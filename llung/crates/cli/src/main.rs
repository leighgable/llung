// crates/cli/src/main.rs
use crate::llung_core::{
    app::{CoreConfig, LlungApp},
    identity::{
        being::create_local_being_interactive,
        session::{ensure_profile_dir, list_profiles, profile_db_path},
    },
    network::NetworkCommand,
};
use std::path::PathBuf;
use tokio::io::{self, AsyncBufReadExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Pick or create profile (interactive)
    let db_path = run_cli_session().await?;

    // 2. Start core
    let (app, mut event_rx) = LlungApp::init(CoreConfig {
        db_path,
        agent: None, // or parse args
        chat_topic: "introductions".into(),
    })
    .await?;

    let cmd_tx = app.command_tx();
    let mut stdin = io::BufReader::new(io::stdin()).lines();

    // 3. Stdin loop (exactly what you have today)
    loop {
        tokio::select! {
            Ok(Some(line)) = stdin.next_line() => {
                cmd_tx.send(NetworkCommand::PublishMessage {
                    topic: libp2p::gossipsub::IdentTopic::new("introductions"),
                    contents: line.into_bytes(),
                }).await?;
            }
            Some(event) = event_rx.recv() => {
                match event {
                    NetworkEvent::MessageReceived { topic, sender, data } => {
                        println!("[{sender} on {topic}]: {}", String::from_utf8_lossy(&data));
                    }
                    NetworkEvent::MediaCidFound { cid, providers } => {
                        println!("Providers for {cid}: {providers:?}");
                    }
                    _ => {}
                }
            }
        }
    }
}
