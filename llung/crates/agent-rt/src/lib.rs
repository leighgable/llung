use std::path::PathBuf;
use tokio::sync::{mpsc, oneshot};
use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store};

mod kernel;
mod security;
mod verification;

// host bindings
wasmtime::component::bindgen!({
    path: "wit/tool.wit",
    world: "universal-tool",
});

/// message contract sent from UI to the core worker
pub struct ToolRequest {
    pub wasm_path: PathBuf,
    pub arguments_json: String,
    pub response_tx: oneshot::Sender<Result<String, String>>,
}

/// initialization function called by main application setup
pub fn spawn_wasm_worker() -> mpsc::Sender<ToolRequest> {
    let (tx, mut rx) = mpsc::channel::<ToolRequest>(32);

    // spawn the long-running worker onto Tokio's asynchronous thread pool
    tokio::spawn(async move {
        // initialize the Wasm engine once outside the hot execution loop
        let mut config = Config::new();
        config.wasm_component_model(true);

        let engine = Engine::new(&config).expect("Failed to create Wasm engine");
        let linker = Linker::new(&engine);

        tracing::info!("Wasm Background Worker Active.");

        // loop indefinitely, waiting for UI execution requests
        while let Some(request) = rx.recv().await {
            let engine_clone = engine.clone();
            let linker_clone = linker.clone();

            tokio::spawn(async move {
                let result = run_tool(
                    &engine_clone,
                    &linker_clone,
                    request.wasm_path,
                    request.arguments_json,
                )
                .await;

                // send the final result back to the specific UI sender
                let _ = request.response_tx.send(result);
            });
        }
    });

    tx // return the sending handle back to the UI
}

/// isolates, instantiates, runs, and tears down the Wasm module
async fn run_tool(
    engine: &Engine,
    linker: &Linker<()>,
    wasm_path: PathBuf,
    args: String,
) -> Result<String, String> {
    let mut store = Store::new(engine, ());

    // load the component dynamically from disk
    let component = Component::from_file(engine, &wasm_path)
        .map_err(|e| format!("Failed to load Wasm file: {}", e))?;

    // instantiate inside the isolated store sandbox
    let (bindings, _) = StandardTool::instantiate_async(&mut store, &component, linker)
        .await
        .map_err(|e| format!("Wasm Sandbox Initialization Failed: {}", e))?;

    // Call the dynamic tool hook safely
    let output = bindings
        .call_execute(&mut store, &args)
        .await
        .map_err(|e| format!("Wasm Execution Runtime Error: {}", e))?;

    Ok(output)
}
