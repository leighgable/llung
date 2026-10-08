// inside agent-rt/src/security.rs
use crate::aukora::kernel::{Grant, KernelState, Manifest, Receipt, compute_sha256_bytes};
use agent_rt::ToolRequest;
use tokio::sync::mpsc;

pub async fn secure_execute_agent_tool(
    state: &mut KernelState,
    wasm_tx: &mpsc::Sender<ToolRequest>,
    manifest: Manifest,
    grant: Grant,
) -> Result<Receipt, String> {
    // through the Aukora logic first
    // checks anti-replay (nonces) and verifies key pins
    let mut receipt = state
        .consume_manifest_use_core(manifest.clone(), grant)
        .map_err(|e| format!("Aukora Guard Denied Execution: {}", e))?;

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let request = ToolRequest {
        wasm_path: format!("runtime-tools/{}.wasm", manifest.target_effect).into(),
        arguments_json: String::from_utf8_lossy(&manifest.arguments).into_owned(),
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

    Ok(receipt)
}
