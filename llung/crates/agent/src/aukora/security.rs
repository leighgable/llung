// inside agent-rt/src/security.rs
use crate::aukora::kernel::{Grant, KernelState, Manifest, Receipt, compute_sha256_bytes};
use agent_rt::ToolRequest;
use tokio::sync::mpsc;

/// Runs the effect through the Aukora guard and the wasm worker.
/// Returns the signed receipt *and* the tool's raw output — the caller
/// decides what to publish (chat wants the output, audit wants the
/// receipt).
/// Where wasm tools live: see [`crate::paths::tools_dir`] (env
/// override, next to the executable, then the source tree — never
/// dependent on the launch directory).
fn tool_wasm_path(effect: &str) -> std::path::PathBuf {
    crate::paths::tool_wasm_path(effect)
}

/// Tier::Host effects (e.g. `host:show_image`) execute in the
/// ORCHESTRATOR, not here — they need `AgentLoopContext` (the agent's
/// network stack, media cache). After the orchestrator performed the
/// effect and holds the tool's output string, this runs the SAME kernel
/// gate as every other tier and commits the receipt to that output, so
/// the audit trail is identical to wasm/host/bwrap executors.
pub fn commit_host_effect(
    state: &mut KernelState,
    manifest: Manifest,
    grant: Grant,
    output: &str,
) -> Result<Receipt, String> {
    let mut receipt = state
        .consume_manifest_use_core(manifest.clone(), grant)
        .map_err(|e| format!("Aukora Guard Denied Execution: {}", e))?;
    receipt.output_commitment = compute_sha256_bytes(output.as_bytes());
    state.advance_history_root(&receipt);
    Ok(receipt)
}

pub async fn secure_execute_agent_tool(
    state: &mut KernelState,
    wasm_tx: &mpsc::Sender<ToolRequest>,
    manifest: Manifest,
    grant: Grant,
) -> Result<(Receipt, String), String> {
    let places = crate::places::current();
    secure_execute_agent_tool_in(state, wasm_tx, manifest, grant, &places).await
}

/// [`secure_execute_agent_tool`] against an explicit places snapshot:
/// the agent loop passes the one its turn started with, so every tool
/// in a turn sees the same workspace / runs / data directories.
pub async fn secure_execute_agent_tool_in(
    state: &mut KernelState,
    wasm_tx: &mpsc::Sender<ToolRequest>,
    manifest: Manifest,
    grant: Grant,
    places: &crate::places::Places,
) -> Result<(Receipt, String), String> {
    // through the Aukora logic first
    // checks anti-replay (nonces) and verifies key pins
    let mut receipt = state
        .consume_manifest_use_core(manifest.clone(), grant)
        .map_err(|e| format!("Aukora Guard Denied Execution: {}", e))?;

    let arguments_json = String::from_utf8_lossy(&manifest.arguments).into_owned();

    // Tier-based dispatch (see effects.rs): every tier funnels through
    // the kernel gate above and gets the same receipt + merkle
    // treatment below; only the executor differs.
    let effect = crate::effects::classify(&manifest.target_effect);
    match effect.tier {
        crate::effects::Tier::Wasm => {
            // fall through to the wasm worker below
        }
        crate::effects::Tier::FsRead if effect.name == "pdf_read" => {
            // Read-only, but a real process: the llung-pdf helper runs
            // inside bubblewrap (PDFium/ONNX Runtime never in-process).
            let output = crate::sandbox::pdf::run(&arguments_json, places)
                .await
                .map_err(|e| format!("pdf_read failed: {e}"))?;
            receipt.output_commitment = compute_sha256_bytes(output.as_bytes());
            state.advance_history_root(&receipt);
            return Ok((receipt, output));
        }
        crate::effects::Tier::Net => {
            // Host-side HTTP with SSRF checks (web.rs). The operator
            // approved this URL (or holds a standing grant) upstream.
            // Only the one known net effect: `net:<anything>` must not
            // become a fetch under another name (and another grant).
            let output = match effect.name {
                "web_fetch" => crate::web::run(&arguments_json, places)
                    .await
                    .map_err(|e| format!("Fetch failed: {e}"))?,
                "web_download" => crate::web::download(&arguments_json, places)
                    .await
                    .map_err(|e| format!("Download failed: {e}"))?,
                _ => {
                    return Err(format!(
                        "effect '{}' (tier 'net') has no executor — only web_fetch / web_download",
                        manifest.target_effect
                    ));
                }
            };
            receipt.output_commitment = compute_sha256_bytes(output.as_bytes());
            state.advance_history_root(&receipt);
            return Ok((receipt, output));
        }
        crate::effects::Tier::FsRead | crate::effects::Tier::Edit => {
            // Host-native tools execute in-process (no wasm, no
            // external processes).
            let Some(host_result) = crate::tools::host::dispatch(effect.name, &arguments_json, Some(places))
            else {
                return Err(format!(
                    "no host executor for effect '{}' (tier {})",
                    manifest.target_effect,
                    effect.tier.as_str()
                ));
            };
            let output = host_result.map_err(|e| format!("Host tool failed: {e}"))?;
            receipt.output_commitment = compute_sha256_bytes(output.as_bytes());
            state.advance_history_root(&receipt);
            return Ok((receipt, output));
        }
        crate::effects::Tier::ProcBash => {
            // bubblewrap-sandboxed shell (proc:bash). Workspace is ro by
            // default; the receipt commits to the captured output.
            let output = crate::sandbox::bash::run(&arguments_json, places)
                .await
                .map_err(|e| format!("Bash sandbox failed: {e}"))?;
            receipt.output_commitment = compute_sha256_bytes(output.as_bytes());
            state.advance_history_root(&receipt);
            return Ok((receipt, output));
        }
        crate::effects::Tier::Repl => {
            // Stateful sandboxed interpreter session (python,
            // chibi-scheme, futhark c backend).
            let output = crate::sandbox::repl::run(&manifest.agent_id, &arguments_json, places)
                .await
                .map_err(|e| format!("Repl failed: {e}"))?;
            receipt.output_commitment = compute_sha256_bytes(output.as_bytes());
            state.advance_history_root(&receipt);
            return Ok((receipt, output));
        }
        crate::effects::Tier::Host => {
            // Host effects need AgentLoopContext (network stack, db) —
            // the orchestrator executes them and funnels the result
            // through `commit_host_effect`. Reaching this arm means a
            // routing bug, not a missing grant.
            return Err(format!(
                "effect '{}' (tier 'host') executes in the orchestrator — \
                 route it through commit_host_effect",
                manifest.target_effect
            ));
        }
        crate::effects::Tier::Gpu => {
            // GPU-backed execution (futhark opencl). Phase 5 adds the
            // human grant prompt; the kernel gate above already ran.
            let output =
                crate::sandbox::repl::run_gpu(&manifest.agent_id, effect.name, &arguments_json, places)
                    .await
                    .map_err(|e| format!("Gpu effect failed: {e}"))?;
            receipt.output_commitment = compute_sha256_bytes(output.as_bytes());
            state.advance_history_root(&receipt);
            return Ok((receipt, output));
        }
        tier => {
            return Err(format!(
                "effect '{}' (tier '{}') has no executor yet — lands in a later phase",
                manifest.target_effect,
                tier.as_str()
            ));
        }
    }

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let request = ToolRequest {
        wasm_path: tool_wasm_path(&manifest.target_effect),
        arguments_json,
        response_tx: resp_tx,
    };

    wasm_tx
        .send(request)
        .await
        .map_err(|e| format!("WASM worker unreachable: {}", e))?;

    let wasm_output = resp_rx
        .await
        .map_err(|e| format!("Wasm worker dropped: {}", e))?
        .map_err(|e| format!("Wasm Execution Failed: {}", e))?;

    // tools result to the cryptographic receipt
    receipt.output_commitment = compute_sha256_bytes(wasm_output.as_bytes());

    // internal merkle state now that the output is committed
    state.advance_history_root(&receipt);

    Ok((receipt, wasm_output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aukora::kernel::{KernelState, Manifest, ReceiptStatus, compute_sha256, compute_sha256_bytes};
    use crate::aukora::supervisor::SupervisorKey;

    /// Kernel pinned to a fresh supervisor, plus the key to sign grants.
    fn kernel_with_supervisor() -> (KernelState, SupervisorKey) {
        let supervisor = SupervisorKey::generate();
        let mut kernel = KernelState::default();
        kernel.pinned_keys.insert(supervisor.public_pin());
        (kernel, supervisor)
    }

    fn signed_manifest(effect: &str, args: &str, nonce: u64, supervisor: &SupervisorKey) -> (Manifest, crate::aukora::kernel::Grant) {
        let manifest = Manifest {
            agent_id: "test-agent".into(),
            target_effect: effect.into(),
            arguments: args.as_bytes().to_vec(),
            nonce,
        };
        let hash = compute_sha256(&manifest);
        let grant = crate::aukora::kernel::Grant {
            manifest_hash: hash,
            authorized_by: supervisor.public_pin(),
            signature: supervisor.sign(&hash),
        };
        (manifest, grant)
    }

    #[tokio::test]
    async fn host_tier_executes_in_process_through_the_kernel_gate() {
        let dir = std::env::temp_dir().join(format!(
            "llung-sec-glob-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("found.rs"), "fn found() {}").unwrap();

        let (mut kernel, supervisor) = kernel_with_supervisor();
        let (wasm_tx, _rx) = mpsc::channel(1);
        let args = serde_json::json!({
            "pattern": "**/*.rs",
            "path": dir.to_string_lossy(),
        })
        .to_string();

        // Bare name classifies to FsRead and runs in-process.
        let (manifest, grant) = signed_manifest("glob", &args, 1, &supervisor);
        // The dir is the agent's workspace: host tools are confined to
        // the places (an outside root is refused — see the test below).
        let places = crate::places::Places::fixed(&dir, dir.join("runs"), vec![]);
        let (receipt, output) =
            secure_execute_agent_tool_in(&mut kernel, &wasm_tx, manifest.clone(), grant, &places)
                .await
                .expect("host-tier execution");

        assert_eq!(receipt.status, ReceiptStatus::Executed);
        assert_eq!(receipt.manifest_hash, compute_sha256(&manifest));
        assert_eq!(
            receipt.output_commitment,
            compute_sha256_bytes(output.as_bytes()),
            "receipt commits to the tool output"
        );
        assert!(output.contains("found.rs"), "got: {output}");
        // The merkle root advanced past the genesis root.
        assert_ne!(kernel.current_history_root, [0u8; 32]);

        // Anti-replay still applies on the host path: same nonce again fails.
        let hash = compute_sha256(&manifest);
        let grant2 = crate::aukora::kernel::Grant {
            manifest_hash: hash,
            authorized_by: supervisor.public_pin(),
            signature: supervisor.sign(&hash),
        };
        let err = secure_execute_agent_tool_in(&mut kernel, &wasm_tx, manifest, grant2, &places)
            .await
            .unwrap_err();
        assert!(err.contains("Replay"), "got: {err}");

        // Agent host tools are confined: a root outside the places is
        // refused (the operator's own tools are not).
        let elsewhere = crate::places::Places::fixed(dir.join("runs"), dir.join("runs2"), vec![]);
        std::fs::create_dir_all(dir.join("runs")).unwrap();
        let (manifest, grant) = signed_manifest("glob", &args, 2, &supervisor);
        let err = secure_execute_agent_tool_in(&mut kernel, &wasm_tx, manifest, grant, &elsewhere)
            .await
            .unwrap_err();
        assert!(err.contains("outside what this agent may read"), "got: {err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn explicit_prefix_classifies_to_the_same_tier() {
        let dir = std::env::temp_dir().join(format!(
            "llung-sec-fsglob-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("x.txt"), "hi").unwrap();

        let (mut kernel, supervisor) = kernel_with_supervisor();
        let (wasm_tx, _rx) = mpsc::channel(1);
        let args = serde_json::json!({
            "pattern": "**/*.txt",
            "path": dir.to_string_lossy(),
        })
        .to_string();
        let (manifest, grant) = signed_manifest("fs:glob", &args, 7, &supervisor);
        let places = crate::places::Places::fixed(&dir, dir.join("runs"), vec![]);
        let (_receipt, output) = secure_execute_agent_tool_in(&mut kernel, &wasm_tx, manifest, grant, &places)
            .await
            .expect("fs:-prefixed glob runs on the host executor");
        assert!(output.contains("x.txt"), "got: {output}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Host effects execute in the orchestrator; the kernel gate must
    /// still commit a receipt to the produced output — same audit
    /// trail as every other tier — and anti-replay must apply.
    #[tokio::test]
    async fn host_effect_commits_receipt_to_the_output() {
        let (mut kernel, supervisor) = kernel_with_supervisor();
        let (manifest, grant) =
            signed_manifest("host:show_image", "{\"path\":\"loss.png\"}", 11, &supervisor);
        let receipt = commit_host_effect(&mut kernel, manifest.clone(), grant, "published image loss.png")
            .expect("host gate commits");
        assert_eq!(receipt.status, ReceiptStatus::Executed);
        assert_eq!(
            receipt.output_commitment,
            compute_sha256_bytes(b"published image loss.png"),
            "receipt commits to the orchestrator-produced output"
        );
        assert_ne!(kernel.current_history_root, [0u8; 32]);

        // Anti-replay applies exactly like the in-process executors.
        let hash = compute_sha256(&manifest);
        let grant2 = crate::aukora::kernel::Grant {
            manifest_hash: hash,
            authorized_by: supervisor.public_pin(),
            signature: supervisor.sign(&hash),
        };
        let err = commit_host_effect(&mut kernel, manifest, grant2, "again").unwrap_err();
        assert!(err.contains("Replay"), "got: {err}");
    }

    #[tokio::test]
    async fn unimplemented_tiers_fail_cleanly_after_authorization() {
        let (mut kernel, supervisor) = kernel_with_supervisor();
        let (wasm_tx, _rx) = mpsc::channel(1);
        // Unknown gpu: effects have no executor; repl:/proc: do.
        let (manifest, grant) =
            signed_manifest("gpu:rocm", "{\"input\":\"1\"}", 3, &supervisor);
        let err = secure_execute_agent_tool(&mut kernel, &wasm_tx, manifest, grant)
            .await
            .unwrap_err();
        assert!(err.contains("no executor"), "got: {err}");
        assert!(err.contains("gpu:rocm"), "got: {err}");
        // The authorized-but-unexecuted attempt IS recorded in history
        // (consume_manifest_use_core folds the manifest into the root),
        // but no receipt/output commitment exists without an executor.
        assert_ne!(kernel.current_history_root, [0u8; 32]);
    }
}
