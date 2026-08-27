// crates/cli/src/main.rs
use llung_core::{
    app::LlungApp,
    config::CoreConfig,
    identity::{
        being::{BeingKind, create_local_being_interactive},
        session::{ensure_profile_dir, list_identities},
    },
    network::NetworkCommand,
};
use std::io::{self, Write};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, BufReader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = run_cli_session().await?;
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
            agent: None,
            chat_topic: "introductions".into(),
        },
        db,
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

async fn run_cli_session() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let profiles = list_identities();

    if profiles.is_empty() {
        println!("No existing profiles found.");
        return run_cli_registration().await;
    }

    println!("\n=== Llung ===\n");
    println!("Existing profiles:");
    for (i, p) in profiles.iter().enumerate() {
        println!("  {}. {}", i + 1, p.profile_name);
    }
    let next = profiles.len() + 1;
    println!("  {}. Create new profile", next);
    println!("  {}. Exit", next + 1);

    loop {
        print!("\nSelect: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        match input.trim().parse::<usize>() {
            Ok(n) if n >= 1 && n <= profiles.len() => {
                println!("Logging in as '{}'...", profiles[n - 1].profile_name);
                return Ok(profiles[n - 1].db_path.clone());
            }
            Ok(n) if n == next => return run_cli_registration().await,
            Ok(n) if n == next + 1 => std::process::exit(0),
            _ => println!("Invalid option."),
        }
    }
}

async fn run_cli_registration() -> Result<PathBuf, Box<dyn std::error::Error>> {
    print!("Choose a profile name: ");
    io::stdout().flush()?;

    let mut name = String::new();
    io::stdin().read_line(&mut name)?;
    let name = name.trim().to_lowercase();

    if name.is_empty() {
        return Err("Profile name cannot be empty.".into());
    }

    let db_path = ensure_profile_dir(&name)?;
    println!("Created profile '{}'.", name);
    Ok(db_path)
}
