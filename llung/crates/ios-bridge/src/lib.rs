use llung_core::{app::LlungApp, config::CoreConfig, network::NetworkCommand};
use std::sync::Arc;
use tokio::runtime::Runtime;

pub struct IosBridge {
    rt: Runtime,
    app: Arc<LlungApp>,
    // event handling needs a Swift callback or an async stream
}

#[uniffi::export]
fn ios_init(db_path: String) -> Arc<IosBridge> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (app, mut event_rx) = rt.block_on(async {
        LlungApp::init(CoreConfig {
            db_path: db_path.into(),
            agent: None,
            chat_topic: "introductions".into(),
        })
        .await
        .unwrap()
    });

    let bridge = Arc::new(IosBridge {
        rt,
        app: Arc::new(app),
    });

    // Spawn event forwarder to Swift callback
    // (simplified: in practice you'd use a callback interface)

    bridge
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
