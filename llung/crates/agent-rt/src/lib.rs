use std::path::PathBuf;
use tokio::sync::{mpsc, oneshot};
use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store};

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

/// Per-invocation WASI state. Tools are pure compute today, but wasip2
/// components may import WASI interfaces (clock, poll, stdio) via std;
/// granting them here keeps those imports linkable without giving tools
/// any host-application state.
struct ToolWasi {
    ctx: wasmtime_wasi::WasiCtx,
    table: wasmtime_wasi::ResourceTable,
}

impl wasmtime_wasi::WasiView for ToolWasi {
    fn ctx(&mut self) -> wasmtime_wasi::WasiCtxView<'_> {
        wasmtime_wasi::WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
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
        let mut linker = Linker::<ToolWasi>::new(&engine);
        if let Err(e) = wasmtime_wasi::p2::add_to_linker_sync(&mut linker) {
            tracing::error!("Failed to link WASI p2: {e}");
            return;
        }

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
    linker: &Linker<ToolWasi>,
    wasm_path: PathBuf,
    args: String,
) -> Result<String, String> {
    let mut store = Store::new(
        engine,
        ToolWasi {
            ctx: wasmtime_wasi::WasiCtx::builder().build(),
            table: wasmtime_wasi::ResourceTable::new(),
        },
    );

    // load the component dynamically from disk
    let component = Component::from_file(engine, &wasm_path)
        .map_err(|e| format!("Failed to load Wasm file: {}", e))?;

    // instantiate inside the isolated store sandbox.
    // (bindgen names the world struct after the WIT world: universal-tool.)
    let bindings = UniversalTool::instantiate_async(&mut store, &component, linker)
        .await
        .map_err(|e| format!("Wasm Sandbox Initialization Failed: {}", e))?;

    // Call the dynamic tool hook safely (plain exported func: sync call)
    let output = bindings
        .call_execute(&mut store, &args)
        .map_err(|e| format!("Wasm Execution Runtime Error: {}", e))?;

    output.map_err(|e| format!("Tool execution failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_braille(c: char) -> bool {
        ('\u{2800}'..='\u{28FF}').contains(&c)
    }

    /// End-to-end: run the landscape component through the same host path
    /// the agent's secured tool executor uses. Skips (passes) when the
    /// component hasn't been built yet — see crates/tools/landscape/README.md.
    #[tokio::test]
    async fn landscape_component_executes() {
        let wasm = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../runtime-tools/landscape.wasm");
        if !wasm.exists() {
            eprintln!("{} not built; skipping", wasm.display());
            return;
        }

        let tx = spawn_wasm_worker();
        let (resp_tx, resp_rx) = oneshot::channel();
        tx.send(ToolRequest {
            wasm_path: wasm,
            arguments_json: r#"{"expr": "sin(x)*cos(y)", "width": 40, "height": 16, "grid": 10}"#
                .into(),
            response_tx: resp_tx,
        })
        .await
        .expect("worker channel open");

        let output = resp_rx
            .await
            .expect("worker responded")
            .expect("tool executed");

        assert!(
            output.chars().any(is_braille),
            "expected braille art, got: {output}"
        );
        assert!(output.lines().count() <= 16);
    }

    #[tokio::test]
    async fn missing_component_is_a_clean_error() {
        let tx = spawn_wasm_worker();
        let (resp_tx, resp_rx) = oneshot::channel();
        tx.send(ToolRequest {
            wasm_path: "runtime-tools/does-not-exist.wasm".into(),
            arguments_json: "{}".into(),
            response_tx: resp_tx,
        })
        .await
        .expect("worker channel open");

        let result = resp_rx.await.expect("worker responded");
        assert!(result.is_err(), "missing wasm must error, got {result:?}");
    }
}
