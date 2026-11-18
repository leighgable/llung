use std::io;
use std::path::PathBuf;

/// The single-account session model: one account per user, shared
/// across machines/devices by copying/mounting the account directory.
/// Registration is not a menu anymore — the account is loaded if it
/// exists, created exactly once if it doesn't. The account's display
/// name (and avatar) are collected later by the TUI registration panel
/// or the CLI prompt, at being-creation time.

/// Override of the data directory (tests, portable installs). Defaults
/// to the platform data dir (`~/.local/share/llung` on Linux).
pub fn data_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("LLUNG_DATA_DIR") {
        if !custom.is_empty() {
            return PathBuf::from(custom);
        }
    }
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llung")
}

/// The one and only account database.
pub fn account_db_path() -> PathBuf {
    data_dir().join("account").join("node.db")
}

pub fn account_exists() -> bool {
    account_db_path().exists()
}

/// Create the account directory (the database itself is created when
/// `Database::new` opens the path). Returns the database path.
pub fn ensure_account_dir() -> io::Result<PathBuf> {
    let dir = data_dir().join("account");
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join("node.db"))
}

/// Resolve the account database: load the existing account, adopt a
/// legacy single profile, or register a fresh one. Only the last case
/// writes anything, and it writes exactly one account.
pub fn resolve_account_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if account_exists() {
        return Ok(account_db_path());
    }
    if migrate_legacy_profile()? {
        return Ok(account_db_path());
    }
    let db_path = ensure_account_dir()?;
    println!("✅ New account registered at {}", db_path.display());
    Ok(db_path)
}

/// Pre-single-account layouts stored identities under `profiles/<name>/`.
/// If exactly that exists, adopt the most recently used profile (the
/// newest `node.db` by modification time) as THE account, moved into
/// place so nothing is copied and the old path stops existing.
pub fn migrate_legacy_profile() -> io::Result<bool> {
    let profiles = data_dir().join("profiles");
    let mut candidates: Vec<(PathBuf, PathBuf)> = Vec::new(); // (dir, node.db)
    if profiles.is_dir() {
        for entry in std::fs::read_dir(&profiles)? {
            let dir = entry?.path();
            let db = dir.join("node.db");
            if db.is_file() {
                candidates.push((dir, db));
            }
        }
    }
    if candidates.is_empty() {
        return Ok(false);
    }
    // Newest first.
    candidates.sort_by_key(|(_, db)| {
        std::fs::metadata(db)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH)
    });
    candidates.reverse();
    let (dir, db) = &candidates[0];
    if candidates.len() > 1 {
        println!(
            "Note: adopting '{}' (most recently used) as the account; other legacy profiles remain under profiles/.",
            dir.display()
        );
    }
    let target = data_dir().join("account");
    std::fs::create_dir_all(data_dir())?;
    match std::fs::rename(dir, &target) {
        Ok(()) => {
            println!("Migrated legacy profile '{}' to the account.", db.display());
            Ok(true)
        }
        Err(e) => {
            // Typically EXDEV (crosses filesystems): fall back to a copy.
            copy_dir(dir, &target)?;
            std::fs::remove_dir_all(dir)?;
            println!("Migrated legacy profile '{}' to the account (copied).", db.display());
            let _ = e;
            Ok(true)
        }
    }
}

fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LLUNG_DATA_DIR is process-global; env-touching tests in this
    /// binary must not run concurrently.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct DataDirGuard {
        previous: Option<String>,
        dir: PathBuf,
    }
    impl DataDirGuard {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "llung-session-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let previous = std::env::var("LLUNG_DATA_DIR").ok();
            // (unsafe: env mutation; this is the only test in the
            // binary touching LLUNG_DATA_DIR.)
            #[allow(unused_unsafe)]
            unsafe {
                std::env::set_var("LLUNG_DATA_DIR", &dir);
            }
            Self { previous, dir }
        }
    }
    impl Drop for DataDirGuard {
        fn drop(&mut self) {
            #[allow(unused_unsafe)]
            unsafe {
                match &self.previous {
                    Some(v) => std::env::set_var("LLUNG_DATA_DIR", v),
                    None => std::env::remove_var("LLUNG_DATA_DIR"),
                }
            }
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    fn touch(path: &std::path::Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, b"db").unwrap();
    }

    #[test]
    fn resolves_single_account_and_migrates_legacy() {
        let _lock = ENV_LOCK.lock().unwrap();
        // Fresh dir: registers exactly one account, idempotent after.
        let guard = DataDirGuard::new("fresh");
        let p1 = resolve_account_path().unwrap();
        assert_eq!(p1, account_db_path());
        assert!(
            p1.parent().unwrap().is_dir(),
            "account dir created; Database::new creates the file"
        );
        // Simulate Database::new opening it, then a later run must load.
        touch(&p1);
        let p2 = resolve_account_path().unwrap();
        assert_eq!(p1, p2, "second run must load, not recreate");
        drop(guard);

        // Legacy profiles: the newest node.db is adopted wholesale.
        let guard = DataDirGuard::new("legacy");
        let old = data_dir().join("profiles/old-laptop/node.db");
        let new = data_dir().join("profiles/daily/node.db");
        touch(&old);
        std::thread::sleep(std::time::Duration::from_millis(20));
        touch(&new);
        let resolved = resolve_account_path().unwrap();
        assert_eq!(resolved, account_db_path());
        assert!(resolved.exists());
        assert!(
            !data_dir().join("profiles/daily").exists(),
            "adopted profile dir must have moved"
        );
        assert!(
            data_dir().join("profiles/old-laptop").exists(),
            "other legacy profiles are left alone"
        );
        drop(guard);
    }

    #[test]
    fn no_account_without_legacy_creates_dir_only() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = DataDirGuard::new("empty");
        assert!(!account_exists());
        let path = ensure_account_dir().unwrap();
        assert_eq!(path, account_db_path());
        assert!(path.parent().unwrap().is_dir());
        assert!(!path.exists(), "Database::new creates the file, not session");
    }
}
