// crates/cli/src/main.rs
use llung_core::{
    app::LlungApp,
    config::CoreConfig,
    identity::{
        being::{BeingKind, create_local_being_interactive},
        session::resolve_account_path,
    },
    network::NetworkCommand,
};
use tokio::io::{AsyncBufReadExt, BufReader};

const KEY_USAGE: &str = "\
Usage: llung-cli key <set|get|delete|list> [provider]

Providers name a model backend's API key (e.g. kimi). Keys are stored
in the login keyring (freedesktop Secret Service), encrypted at rest;
LLUNG_<PROVIDER>_API_KEY overrides for one shell.\n
Examples:
  echo \"sk-...\" | llung-cli key set kimi
  llung-cli key get kimi        # prints the key (for scripts)
  llung-cli key delete kimi
  llung-cli key list";

/// Key management runs before any account/network work: it must be
/// usable on a fresh machine with no account yet.
async fn run_key_command(args: &[String]) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(sub) = args.first().map(String::as_str) else {
        return Ok(false);
    };
    if sub != "key" {
        return Ok(false);
    }
    let rest = &args[1..];
    match rest.first().map(String::as_str) {
        Some("set") => {
            let provider = rest.get(1).map(String::as_str).ok_or("key set needs a provider")?;
            let mut key = String::new();
            std::io::stdin().read_line(&mut key)?;
            let key = key.trim();
            if key.is_empty() {
                return Err("empty key — pipe it on stdin: echo \"sk-...\" | llung-cli key set {provider}".into());
            }
            llung_core::secrets::store_api_key(provider, key).await?;
            println!("Stored {provider} API key in the login keyring.");
        }
        Some("get") => {
            let provider = rest.get(1).map(String::as_str).ok_or("key get needs a provider")?;
            match llung_core::secrets::load_api_key(provider).await {
                Ok(key) => println!("{key}"),
                Err(e) => return Err(e.into()),
            }
        }
        Some("delete") => {
            let provider = rest.get(1).map(String::as_str).ok_or("key delete needs a provider")?;
            if llung_core::secrets::delete_api_key(provider).await? {
                println!("Deleted {provider} API key.");
            } else {
                println!("No stored {provider} key.");
            }
        }
        Some("list") => {
            // Providers are discovered on demand; list the known ones.
            for provider in ["kimi"] {
                let state = if llung_core::secrets::has_api_key(provider).await {
                    "set"
                } else {
                    "not set"
                };
                println!("{provider}: {state}");
            }
        }
        _ => {
            eprintln!("{KEY_USAGE}");
            std::process::exit(2);
        }
    }
    Ok(true)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if run_key_command(&args).await? {
        return Ok(());
    }
    // Single account: load it, adopt a legacy profile, or register one.
    let db_path = resolve_account_path()?;
    let db_path_str = db_path.to_str().ok_or("invalid db path")?;

    let (db, machine) = llung_core::storage::db::Database::new(db_path_str)?;
    let being = match db.load_being()? {
        Some(b) => {
            println!("Welcome back, {}.", b.human_name);
            b
        }
        None => create_local_being_interactive(&db, BeingKind::Human)?,
    };

    let (app, mut event_rx) = LlungApp::init(
        CoreConfig {
            db_path,
            chat_topic: "introductions".into(),
            relays: CoreConfig::relays_from_env(),
        },
        machine,
        being,
    )
    .await?;

    let cmd_tx = app.command_tx();
    let mut stdin = BufReader::new(tokio::io::stdin()).lines();

    loop {
        tokio::select! {
            Ok(Some(line)) = stdin.next_line() => {
                cmd_tx.send(NetworkCommand::PublishMessage {
                    topic: libp2p::gossipsub::IdentTopic::new("introductions"),
                    contents: line.into_bytes(),
                }).await?;
            }
            Some(event) = event_rx.recv() => {
                println!("{:?}", event);
            }
        }
    }
}


