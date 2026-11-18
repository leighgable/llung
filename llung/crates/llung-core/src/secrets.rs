// llung-core/src/secrets.rs
//
// API-key storage for cloud model backends (Kimi, ...). The threat
// model and the storage choice, plainly:
//
// - At rest: keys live in the freedesktop Secret Service (gnome-keyring
//   or KWallet) under the user's login session, i.e. encrypted on disk
//   and unlocked only while the user is logged in. This is the
//   strongest practical protection on a desktop without external
//   hardware (TPM/HSM): other users and offline disk access don't get
//   the key.
// - In memory: the key is loaded at agent-spawn time and held by the
//   backend in RAM, like any HTTP client's bearer token. A process
//   running AS the user can read it — no software store can prevent
//   that; only hardware-backed keystore attestation (not wired here)
//   raises that bar.
// - Convenience override: LLUNG_<PROVIDER>_API_KEY env var wins, for
//   CI and throwaway shells (env of a process is not world-readable).
//
// iOS: stubbed (no Secret Service); only the env override works there.

/// Attribute keys every llung secret item carries.
pub const ATTR_SERVICE: &str = "llung";
const ATTR_KIND: (&str, &str) = ("kind", "api-key");

#[derive(Debug)]
pub enum KeyError {
    /// Neither the env override nor the keyring has a key: the hint
    /// says exactly how to configure one.
    NotConfigured { provider: String, hint: String },
    /// The keyring itself failed (no Secret Service, locked, ...).
    Store(String),
}

impl std::fmt::Display for KeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyError::NotConfigured { provider, hint } => {
                write!(f, "no API key for '{provider}' — {hint}")
            }
            KeyError::Store(e) => write!(f, "secret store error: {e}"),
        }
    }
}

impl std::error::Error for KeyError {}

/// `kimi` → `LLUNG_KIMI_API_KEY`.
fn env_var_name(provider: &str) -> String {
    let upper: String = provider
        .chars()
        .map(|c| if c == '-' { '_' } else { c.to_ascii_uppercase() })
        .collect();
    format!("LLUNG_{upper}_API_KEY")
}

fn not_configured(provider: &str) -> KeyError {
    KeyError::NotConfigured {
        provider: provider.to_string(),
        hint: format!(
            "set {} or run `llung-cli key set {provider}` (stored in your login keyring)",
            env_var_name(provider)
        ),
    }
}

/// Load an API key for `provider`: env override first, then the
/// freedesktop Secret Service (default collection).
pub async fn load_api_key(provider: &str) -> Result<String, KeyError> {
    if let Ok(key) = std::env::var(env_var_name(provider)) {
        if !key.is_empty() {
            return Ok(key);
        }
    }
    load_from_keyring(provider).await
}

#[cfg(not(target_os = "ios"))]
fn attributes(provider: &str) -> std::collections::HashMap<&str, &str> {
    [("service", ATTR_SERVICE), ("provider", provider), ATTR_KIND]
        .into_iter()
        .collect()
}

#[cfg(not(target_os = "ios"))]
async fn load_from_keyring(provider: &str) -> Result<String, KeyError> {
    use secret_service::SecretService;
    let service = SecretService::connect(secret_service::EncryptionType::Dh)
        .await
        .map_err(|e| KeyError::Store(e.to_string()))?;
    let collection = service
        .get_default_collection()
        .await
        .map_err(|e| KeyError::Store(e.to_string()))?;
    let items = collection
        .search_items(attributes(provider))
        .await
        .map_err(|e| KeyError::Store(e.to_string()))?;
    let Some(item) = items.into_iter().next() else {
        return Err(not_configured(provider));
    };
    let secret = item
        .get_secret()
        .await
        .map_err(|e| KeyError::Store(e.to_string()))?;
    String::from_utf8(secret).map_err(|_| KeyError::Store("key is not valid UTF-8".into()))
}

#[cfg(target_os = "ios")]
async fn load_from_keyring(provider: &str) -> Result<String, KeyError> {
    Err(not_configured(provider))
}

/// Store (or replace) an API key in the login keyring.
#[cfg(not(target_os = "ios"))]
pub async fn store_api_key(provider: &str, key: &str) -> Result<(), String> {
    use secret_service::SecretService;
    let service = SecretService::connect(secret_service::EncryptionType::Dh)
        .await
        .map_err(|e| e.to_string())?;
    let collection = service.get_default_collection().await.map_err(|e| e.to_string())?;
    collection
        .create_item(
            &format!("llung {provider} API key"),
            attributes(provider),
            key.as_bytes(),
            true, // replace an existing item with the same attributes
            "text/plain",
        )
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(target_os = "ios")]
pub async fn store_api_key(_provider: &str, _key: &str) -> Result<(), String> {
    Err("no secret store on this platform — use the env override".into())
}

/// Delete a stored API key; true when one existed.
#[cfg(not(target_os = "ios"))]
pub async fn delete_api_key(provider: &str) -> Result<bool, String> {
    use secret_service::SecretService;
    let service = SecretService::connect(secret_service::EncryptionType::Dh)
        .await
        .map_err(|e| e.to_string())?;
    let collection = service.get_default_collection().await.map_err(|e| e.to_string())?;
    let items = collection
        .search_items(attributes(provider))
        .await
        .map_err(|e| e.to_string())?;
    let found = !items.is_empty();
    for item in items {
        item.delete().await.map_err(|e| e.to_string())?;
    }
    Ok(found)
}

#[cfg(target_os = "ios")]
pub async fn delete_api_key(_provider: &str) -> Result<bool, String> {
    Ok(false)
}

/// True when a key is configured (env or keyring) — without revealing it.
pub async fn has_api_key(provider: &str) -> bool {
    if std::env::var(env_var_name(provider)).is_ok_and(|k| !k.is_empty()) {
        return true;
    }
    load_from_keyring(provider).await.is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_var_names_per_provider() {
        assert_eq!(env_var_name("kimi"), "LLUNG_KIMI_API_KEY");
        assert_eq!(env_var_name("open-router"), "LLUNG_OPEN_ROUTER_API_KEY");
    }

    #[tokio::test]
    async fn env_override_wins_without_touching_the_keyring() {
        // A provider name with no keyring entry: only the env path can
        // succeed, proving precedence and that the keyring isn't needed.
        let var = env_var_name("llung-test-nonexistent-provider");
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var(&var, "sk-test-value");
        }
        let key = load_api_key("llung-test-nonexistent-provider")
            .await
            .expect("env override must resolve");
        assert_eq!(key, "sk-test-value");
        #[allow(unused_unsafe)]
        unsafe {
            std::env::remove_var(&var);
        }
        // And with the env gone, the same provider is NotConfigured.
        match load_api_key("llung-test-nonexistent-provider").await {
            Err(KeyError::NotConfigured { provider, hint }) => {
                assert_eq!(provider, "llung-test-nonexistent-provider");
                assert!(hint.contains("LLUNG_LLUNG_TEST_NONEXISTENT_PROVIDER_API_KEY"));
            }
            other => panic!("expected NotConfigured, got {other:?}"),
        }
    }

    /// Real keyring round-trip. Opt-in (LLUNG_KEYRING_TEST=1): it
    /// briefly writes to the logged-in user's keyring, which CI/headless
    /// boxes may not have.
    #[tokio::test]
    async fn keyring_round_trip_when_opted_in() {
        if std::env::var("LLUNG_KEYRING_TEST").as_deref() != Ok("1") {
            eprintln!("LLUNG_KEYRING_TEST not set; skipping keyring round-trip");
            return;
        }
        let provider = "llung-test-provider";
        store_api_key(provider, "sk-roundtrip")
            .await
            .expect("store (is a Secret Service available?)");
        assert!(has_api_key(provider).await);
        assert_eq!(load_api_key(provider).await.unwrap(), "sk-roundtrip");
        assert!(delete_api_key(provider).await.unwrap());
        assert!(!has_api_key(provider).await);
    }
}
