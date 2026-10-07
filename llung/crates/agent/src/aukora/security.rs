// inside agent-rt/src/security.rs
use crate::kernel::{Grant, KernelState, Manifest, Receipt};

pub fn secure_execute_agent_tool(
    state: &mut KernelState,
    wasm_rt: &WasmRuntime,
    manifest: Manifest,
    grant: Grant,
) -> Result<Receipt, String> {
    // 1. Process the request through the Aukora logic first
    // This checks anti-replay (nonces) and verifies key pins
    let mut receipt = state
        .consume_manifest_use_core(manifest.clone(), grant)
        .map_err(|e| format!("Aukora Guard Denied Execution: {}", e))?;

    // 2. If valid, unpack arguments and pass them into your Wasmtime runtime
    let tool_name = &manifest.target_effect;
    let payload = &manifest.arguments;

    let wasm_output = wasm_rt
        .call_tool(tool_name, payload)
        .map_err(|e| format!("Wasm Execution Failed: {}", e))?;

    // 3. Commit the tool's result to the cryptographic receipt
    receipt.output_commitment = crypto::compute_sha256(&wasm_output);

    // 4. Update internal Merkle state now that the output is committed
    state.advance_history_root(&receipt);

    Ok(receipt)
}
