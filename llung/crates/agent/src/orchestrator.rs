use agent_rt::ToolRequest;
use llung_core::network::{NetworkCommand, NetworkEvent};

pub async fn agent_loop(
    backend: impl LlmBackend,
    wasm_tx: mpsc::Sender<ToolRequest>,
    // ...
) {
    let response = backend.chat(&messages).await?;
    
    if let Some(tool_call) = response.tool_call {
        let result = wasm_tx.send(ToolRequest {
            wasm_path: format!("runtime-tools/generated-src/{}/tool.wasm", tool_call.name),
            arguments_json: tool_call.arguments,
            response_tx,
        }).await?;
        // feed result back into messages as a "tool" role
    }
}
