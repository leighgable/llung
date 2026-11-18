use llung_core::{
    app::LlungApp, config::CoreConfig, network::NetworkCommand, storage::db::Database,
};
use std::sync::Arc;
use tokio::runtime::Runtime;

uniffi::setup_scaffolding!();

#[derive(uniffi::Object)]
pub struct IosBridge {
    rt: Runtime,
    app: Arc<LlungApp>,
    // event handling needs a Swift callback or an async stream
}

#[uniffi::export]
fn ios_init(db_path: String) -> std::result::Result<Arc<IosBridge>, String> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    let (app, _event_rx) = rt.block_on(async {
        let (db, machine) = Database::new(&db_path).map_err(|e| e.to_string())?;
        let being = db
            .load_being()
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "no being registered in this database".to_string())?;
        LlungApp::init(
            CoreConfig {
                db_path: db_path.into(),
                chat_topic: "introductions".into(),
                relays: Vec::new(),
            },
            machine,
            being,
        )
        .await
        .map_err(|e| e.to_string())
    })?;

    Ok(Arc::new(IosBridge {
        rt,
        app: Arc::new(app),
    }))
}

#[uniffi::export]
fn ios_send_message(bridge: Arc<IosBridge>, text: String) {
    let tx = bridge.app.command_tx();
    bridge.rt.spawn(async move {
        let _ = tx
            .send(NetworkCommand::PublishMessage {
                topic: libp2p::gossipsub::IdentTopic::new("introductions"),
                contents: text.into_bytes(),
            })
            .await;
    });
}
