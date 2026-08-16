use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct LocalIdentity {
    pub profile_name: String,
    pub db_path: PathBuf,
}

#[derive(Debug)]
pub enum SessionChoice {
    Login { db_path: PathBuf },
    Register { db_path: PathBuf },
}

fn data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llung")
}

fn list_identities() -> Vec<LocalIdentity> {
    let profiles_dir = data_dir().join("profiles");
    let mut identities = Vec::new();

    if let Ok(entries) = fs::read_dir(&profiles_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let db_file = path.join("node.db");
                if db_file.exists() {
                    let profile_name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    identities.push(LocalIdentity {
                        profile_name,
                        db_path: db_file,
                    });
                }
            }
        }
    }
    identities
}

pub fn prompt_session() -> Result<SessionChoice, Box<dyn std::error::Error>> {
    let profiles_dir = data_dir().join("profiles");
    fs::create_dir_all(&profiles_dir)?;

    let identities = list_identities();

    if identities.is_empty() {
        println!("👤 No existing identities found.");
        return run_registration();
    }

    println!("\n╔════════════════════════════════════╗");
    println!("║       Welcome to Llung             ║");
    println!("╚════════════════════════════════════╝\n");

    println!("Existing identities:");
    for (i, id) in identities.iter().enumerate() {
        println!("  {}. {}", i + 1, id.profile_name);
    }
    let next = identities.len() + 1;
    println!("  {}. ➕ Create new identity", next);
    println!("  {}. 🚪 Exit", next + 1);

    loop {
        print!("\nSelect an option: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        match input.trim().parse::<usize>() {
            Ok(n) if n >= 1 && n <= identities.len() => {
                println!("🔓 Logging in as '{}'...", identities[n - 1].profile_name);
                return Ok(SessionChoice::Login {
                    db_path: identities[n - 1].db_path.clone(),
                });
            }
            Ok(n) if n == next => return run_registration(),
            Ok(n) if n == next + 1 => std::process::exit(0),
            _ => println!("Invalid option, please try again."),
        }
    }
}

fn run_registration() -> Result<SessionChoice, Box<dyn std::error::Error>> {
    println!("\n━━━ New Identity Registration ━━━\n");

    print!("Choose a profile name (e.g., 'alice', 'work'): ");
    io::stdout().flush()?;
    let mut profile_name = String::new();
    io::stdin().read_line(&mut profile_name)?;
    let profile_name = profile_name.trim().to_lowercase();

    if profile_name.is_empty() {
        return Err("Profile name cannot be empty.".into());
    }

    let profile_dir = data_dir().join("profiles").join(&profile_name);
    if profile_dir.exists() {
        return Err(format!(
            "Profile '{}' already exists. Login instead, or pick a new name.",
            profile_name
        )
        .into());
    }

    fs::create_dir_all(&profile_dir)?;
    let db_path = profile_dir.join("node.db");

    println!("✅ Profile '{}' registered.", profile_name);

    Ok(SessionChoice::Register { db_path })
}
