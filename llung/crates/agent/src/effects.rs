// crates/agent/src/effects.rs
//
// The Aukora effect namespace. The kernel's `target_effect` is no longer
// "whatever .wasm file happens to exist": it classifies into a tier, and
// each tier has a policy (does it need a human grant? may it be granted
// perpetually?) and an executor (wasm worker, host tools, and — in later
// phases — the bubblewrap sandbox and the REPL manager).
//
//   tier        prefix     examples                executor     prompt?
//   ──────────  ─────────  ──────────────────────  ───────────  ──────
//   Wasm        (bare)     landscape, plot_device  wasm worker  no
//   Host        host:      show_image              orchestrator no
//   FsRead      fs:        glob, grep              host::       no
//                          pdf_read                bwrap        no (read-only, no network)
//   Edit        edit:      edit                    host::       no
//   ProcBash    proc:      bash (Phase 3)          bwrap        YES
//   Repl        repl:      python, chibi, futhark  REPL mgr     YES
//   Gpu         gpu:       futhark, torch          bwrap+GPU    YES
//   Net         net:       web_fetch, web_download host HTTP    YES
//
// Bare (prefix-less) names resolve through the registry: known host-tool
// names map to their tier, everything else stays Wasm (preserving the
// original behavior where any name maps to `runtime-tools/<name>.wasm`
// and the worker errors if the component is missing).

/// Which executor and policy an effect belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Pure-compute wasm components in `runtime-tools/`. Sandboxed by
    /// construction; no prompt.
    Wasm,
    /// Read-only filesystem inspection (glob, grep). Workspace-only,
    /// no prompt.
    FsRead,
    /// Exact-match file editing (edit). Writes, but verified and
    /// diff-returned; no prompt (Phase 5 may re-decide).
    Edit,
    /// Arbitrary shell commands via the bubblewrap sandbox (Phase 3).
    /// Always prompts.
    ProcBash,
    /// Long-running interpreter sessions (Phase 4). Prompts once per
    /// session.
    Repl,
    /// GPU-backed execution (futhark opencl/cuda, torch). Prompts
    /// unless a standing grant names the exact effect (`gpu:torch`).
    Gpu,
    /// Outbound network access from the host (web_fetch). Always
    /// prompts (the URL is in the prompt): fetched text is untrusted
    /// input, and a URL is also a way OUT — whatever the agent puts in
    /// it reaches the server. Standing grants are allowed.
    Net,
    /// Side-effecting host tools that need the AGENT's context (network
    /// stack, media cache) — e.g. `show_image` publishes a media
    /// message from the agent's own being. The orchestrator executes
    /// them (secure_execute_agent_tool has no such context) and commits
    /// the receipt through the same kernel gate. Publishing to a topic
    /// the agent already posts to is not privileged: no prompt.
    Host,
}

impl Tier {
    /// Short stable identifier, used in receipts and grant rules.
    pub fn as_str(&self) -> &'static str {
        match self {
            Tier::Wasm => "wasm",
            Tier::FsRead => "fs",
            Tier::Edit => "edit",
            Tier::ProcBash => "proc",
            Tier::Repl => "repl",
            Tier::Gpu => "gpu",
            Tier::Host => "host",
            Tier::Net => "net",
        }
    }

    /// Does this tier need a human grant prompt before execution?
    /// Phase 5 wires this into the TUI permission flow; Phase 3/4
    /// executors should also refuse to run unprompted when true.
    pub fn requires_prompt(&self) -> bool {
        matches!(self, Tier::ProcBash | Tier::Repl | Tier::Gpu | Tier::Net)
    }

    /// May the prompt offer "always" (a standing grant) for this tier?
    /// Every tier, Gpu included: a standing grant names one exact effect
    /// (`gpu:torch`, never `gpu:*`), the prompt shows that effect, and
    /// research agents need unattended GPU access. `/grant revoke` undoes it.
    pub fn standing_allowed(&self) -> bool {
        true
    }
}

/// A classified effect: its tier and the bare tool name (prefix stripped).
#[derive(Debug)]
pub struct Effect<'a> {
    pub tier: Tier,
    pub name: &'a str,
}

/// Host-tool names known to the registry (also in `tools::host::HOST_TOOLS`
/// and the tool catalog — keep all three in sync).
/// Bare-name registry: host tools and the sandboxed tools. `bash` runs
/// through bubblewrap (Phase 3) — never in-process like the host tools.
const BARE_TIERS: &[(&str, Tier)] = &[
    ("show_image", Tier::Host),
    ("glob", Tier::FsRead),
    ("grep", Tier::FsRead),
    // Read-only and network-less, but a real process: PDFium + ONNX
    // Runtime run inside bubblewrap (sandbox/pdf.rs), never in-process.
    ("pdf_read", Tier::FsRead),
    ("web_fetch", Tier::Net),
    ("web_download", Tier::Net),
    ("edit", Tier::Edit),
    ("bash", Tier::ProcBash),
    ("repl", Tier::Repl),
];

/// Classify a `target_effect` string into its tier and bare name.
///
/// Explicit prefixes (`fs:glob`, `gpu:futhark-opencl`) win. Bare names
/// consult the host-tool registry, then fall back to Wasm — the catch-all
/// keeps every existing wasm tool working unchanged.
pub fn classify(target_effect: &str) -> Effect<'_> {
    if let Some((prefix, name)) = target_effect.split_once(':') {
        let tier = match prefix {
            "wasm" => Tier::Wasm,
            "fs" => Tier::FsRead,
            "edit" => Tier::Edit,
            "proc" => Tier::ProcBash,
            "repl" => Tier::Repl,
            "gpu" => Tier::Gpu,
            "host" => Tier::Host,
            "net" => Tier::Net,
            // Unknown prefix: do not silently run it as something else.
            // Classify as Wasm under the full original string so the
            // worker reports the missing component with its real name.
            _ => return Effect { tier: Tier::Wasm, name: target_effect },
        };
        return Effect { tier, name };
    }
    for &(name, tier) in BARE_TIERS {
        if target_effect == name {
            return Effect { tier, name };
        }
    }
    Effect { tier: Tier::Wasm, name: target_effect }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_wasm_names_stay_wasm() {
        for name in ["landscape", "artist", "diagram", "plot_device", "image_mask_resize"] {
            let e = classify(name);
            assert_eq!(e.tier, Tier::Wasm, "{name}");
            assert_eq!(e.name, name);
        }
    }

    #[test]
    fn bare_host_tool_names_classify_to_their_tiers() {
        assert_eq!(classify("glob").tier, Tier::FsRead);
        assert_eq!(classify("grep").tier, Tier::FsRead);
        assert_eq!(classify("edit").tier, Tier::Edit);
        assert_eq!(classify("edit").name, "edit");
        // The sandboxed shell is NOT a host tool: it must route to bwrap.
        assert_eq!(classify("bash").tier, Tier::ProcBash);
        assert_eq!(classify("proc:bash").tier, Tier::ProcBash);
        assert_eq!(classify("pdf_read").tier, Tier::FsRead);
        assert_eq!(classify("web_fetch").tier, Tier::Net);
        assert_eq!(classify("net:web_fetch").name, "web_fetch");
        assert_eq!(classify("web_download").tier, Tier::Net);
        assert!(Tier::Net.requires_prompt(), "network access always asks");
        assert!(Tier::Net.standing_allowed());
        assert!(!Tier::FsRead.requires_prompt(), "reading a PDF does not");
    }

    #[test]
    fn explicit_prefixes_win_and_strip() {
        let e = classify("fs:glob");
        assert_eq!(e.tier, Tier::FsRead);
        assert_eq!(e.name, "glob");
        assert_eq!(classify("proc:bash").tier, Tier::ProcBash);
        assert_eq!(classify("repl:python").tier, Tier::Repl);
        assert_eq!(classify("repl:futhark").name, "futhark");
        assert_eq!(classify("gpu:futhark-opencl").tier, Tier::Gpu);
        assert_eq!(classify("gpu:torch").tier, Tier::Gpu);
        assert_eq!(classify("gpu:torch").name, "torch");
        assert_eq!(classify("edit:file").tier, Tier::Edit);
        // host: tools are context-dependent orchestrator effects.
        assert_eq!(classify("host:show_image").tier, Tier::Host);
        assert_eq!(classify("host:show_image").name, "show_image");
        assert_eq!(classify("show_image").tier, Tier::Host);
        // Explicit wasm prefix round-trips.
        assert_eq!(classify("wasm:landscape").tier, Tier::Wasm);
        assert_eq!(classify("wasm:landscape").name, "landscape");
    }

    #[test]
    fn unknown_prefix_is_not_misclassified() {
        // An unknown prefix keeps the FULL original string as the wasm
        // name so the failure message names the real effect.
        let e = classify("xyz:send");
        assert_eq!(e.tier, Tier::Wasm);
        assert_eq!(e.name, "xyz:send");
    }

    #[test]
    fn tier_policy_flags() {
        assert!(!Tier::Wasm.requires_prompt());
        assert!(!Tier::FsRead.requires_prompt());
        assert!(!Tier::Edit.requires_prompt());
        assert!(!Tier::Host.requires_prompt(), "publishing to the agent's own topic is not privileged");
        assert!(Tier::ProcBash.requires_prompt());
        assert!(Tier::Repl.requires_prompt());
        assert!(Tier::Gpu.requires_prompt());

        assert!(Tier::Wasm.standing_allowed());
        assert!(Tier::ProcBash.standing_allowed());
        assert!(Tier::Gpu.standing_allowed(), "GPU may be granted perpetually (exact effect)");
    }

    #[test]
    fn tier_names_are_stable_short_strings() {
        assert_eq!(Tier::Wasm.as_str(), "wasm");
        assert_eq!(Tier::Gpu.as_str(), "gpu");
    }
}
