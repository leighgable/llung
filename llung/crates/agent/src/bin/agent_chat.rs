// crates/agent/src/bin/agent_chat.rs
//
// Minimal REPL for chatting with the local agent backend (SmolVLM Q4 on
// llama-server). Every turn — human input and model reply — is persisted
// to the llung SQLite DB as ChatMessages on the "agent-chat" topic, so
// the conversation survives restarts and is visible to the rest of the
// system (DAG history, TUI, orchestrator).
//
// Usage:
//   cargo run -p agent --bin agent_chat [db_path]
//
// Env overrides:
//   LLAMA_BASE_URL  (default http://localhost:9931/v1)
//   LLM_MODEL       (default SmolVLM)

use std::io::{self, Write};

use agent::interface::{Backend, OpenAiCompatible};
use llung_core::{
    identity::being::BeingKind,
    network::message::{ChatMessage, MessageKind},
    storage::db::Database,
};
use tokio::io::AsyncBufReadExt;

const TOPIC: &str = "agent-chat";

/// Query an OpenAI-compatible `/v1/models` and return the first model id.
async fn first_model_at(base_url: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct ModelList {
        data: Vec<ModelEntry>,
    }
    #[derive(serde::Deserialize)]
    struct ModelEntry {
        id: String,
    }

    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let list: ModelList = reqwest::get(url).await.ok()?.json().await.ok()?;
    list.data.into_iter().next().map(|m| m.id)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = std::env::args().nth(1).unwrap_or_else(|| "agent_chat.db".into());
    let base_url = std::env::var("LLAMA_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:9931/v1".into());
    // llama-server identifies models by their full GGUF path, so ask
    // /v1/models rather than guessing a friendly name.
    let model = match std::env::var("LLM_MODEL") {
        Ok(m) => m,
        Err(_) => first_model_at(&base_url).await.unwrap_or_else(|| "SmolVLM".into()),
    };

    let (db, machine) = Database::new(&db_path)?;

    // The agent's being: reuse the one in this DB, or provision an
    // Agent-kind being up front so replies are attributed correctly.
    let being = match db.load_being()? {
        Some(b) => b,
        None => {
            // Name the agent after the model file stem, not the whole path.
            let name = model
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(&model)
                .split('.')
                .next()
                .unwrap_or(&model)
                .to_string();
            db.create_being_for_machine(&machine.peer_id.to_base58(), &name, BeingKind::Agent)?
        }
    };
    let agent_peer: libp2p::PeerId = being.being_id.parse()?;

    let backend = OpenAiCompatible::new(base_url, &model);
    println!(
        "Chatting with '{}' (model: {}). Type /quit to exit. DB: {}",
        being.human_name, model, db_path
    );

    let mut lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let Some(line) = lines.next_line().await? else {
            break; // EOF
        };
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if text == "/quit" || text == "/exit" {
            break;
        }

        // Chain onto the current tip so the DAG stays linear.
        let parent_id = db
            .load_messages_for_topic(TOPIC)?
            .last()
            .map(|m| m.id.clone());

        let user_msg = ChatMessage::new(
            TOPIC.into(),
            machine.peer_id,
            parent_id,
            MessageKind::Human,
            "you".into(),
            text.into(),
        );
        db.save_message(&user_msg)?;

        // Full history for this topic becomes the prompt.
        let mut prompt = vec![ChatMessage::system(format!(
            "You are {}, a helpful assistant. Respond concisely.",
            being.human_name
        ))];
        prompt.extend(db.load_messages_for_topic(TOPIC)?);

        let reply = match backend.chat(&prompt).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("backend error: {e}");
                continue;
            }
        };
        println!("{reply}\n");

        let reply_msg = ChatMessage::new(
            TOPIC.into(),
            agent_peer,
            Some(user_msg.id.clone()),
            MessageKind::Agent,
            being.human_name.clone(),
            reply,
        );
        db.save_message(&reply_msg)?;
    }

    println!("Bye.");
    Ok(())
}
