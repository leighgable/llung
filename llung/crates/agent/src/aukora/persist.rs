// crates/agent/src/aukora/persist.rs
//
// KernelState <-> SQLite glue. The kernel itself is pure in-memory
// state; this module snapshots it per agent being-id so anti-replay
// nonces, pinned supervisor keys, and the receipt-history merkle root
// all survive process restarts.

use crate::aukora::kernel::KernelState;
use llung_core::storage::db::Database;

/// Load a kernel state for `agent_id`. Fresh agents get the default
/// state; returning agents get their consumed nonces, pinned keys,
/// history root, and nonce counter back.
pub fn load_kernel_state(db: &Database, agent_id: &str) -> KernelState {
    let mut state = KernelState::default();
    if let Ok(Some((root, next_nonce))) = db.kernel_load_snapshot(agent_id) {
        state.current_history_root = root;
        state.next_nonce = next_nonce.max(state.next_nonce);
    }
    if let Ok(nonces) = db.kernel_consumed_nonces(agent_id) {
        state.executed_nonces = nonces.into_iter().collect();
    }
    if let Ok(pins) = db.kernel_pinned_keys(agent_id) {
        state.pinned_keys = pins.into_iter().collect();
    }
    if let Ok(rules) = db.kernel_standing(agent_id) {
        state.standing_grants = rules
            .into_iter()
            .map(|(effect, authorized_by)| crate::aukora::kernel::StandingGrant {
                agent_id: agent_id.to_string(),
                effect,
                authorized_by,
            })
            .collect();
    }
    state
}

/// Persist the pieces of `state` that must survive a restart. All
/// writes are idempotent (INSERT OR IGNORE / upsert), so saving after
/// every execution is safe even though consumed-nonces only ever grows.
pub fn save_kernel_state(
    db: &Database,
    agent_id: &str,
    state: &KernelState,
) -> Result<(), String> {
    db.kernel_save_snapshot(agent_id, &state.current_history_root, state.next_nonce)
        .map_err(|e| format!("kernel snapshot save failed: {e}"))?;
    for &nonce in &state.executed_nonces {
        db.kernel_record_nonce(agent_id, nonce)
            .map_err(|e| format!("kernel nonce save failed: {e}"))?;
    }
    for pin in &state.pinned_keys {
        db.kernel_add_pinned_key(agent_id, pin)
            .map_err(|e| format!("kernel pin save failed: {e}"))?;
    }
    // Standing grants are SYNCED (not just inserted): the kernel state
    // is authoritative, so revocations must delete DB rows too.
    db.kernel_clear_standing(agent_id)
        .map_err(|e| format!("kernel standing clear failed: {e}"))?;
    for rule in &state.standing_grants {
        db.kernel_add_standing(agent_id, &rule.effect, &rule.authorized_by)
            .map_err(|e| format!("kernel standing grant save failed: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aukora::kernel::{Grant, Manifest, compute_sha256};
    use crate::aukora::supervisor::SupervisorKey;

    fn temp_db(tag: &str) -> (Database, String) {
        let path = std::env::temp_dir().join(format!(
            "llung-kernel-persist-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db = Database::open(path.to_str().unwrap()).expect("temp db");
        (db, path.to_string_lossy().into_owned())
    }

    fn signed_grant(manifest: &Manifest, supervisor: &SupervisorKey) -> Grant {
        Grant {
            manifest_hash: compute_sha256(manifest),
            authorized_by: supervisor.public_pin(),
            signature: supervisor.sign(&compute_sha256(manifest)),
        }
    }

    #[test]
    fn load_returns_default_for_fresh_agent() {
        let (db, _path) = temp_db("fresh");
        let state = load_kernel_state(&db, "agent-x");
        assert!(state.executed_nonces.is_empty());
        assert!(state.pinned_keys.is_empty());
        assert_eq!(state.current_history_root, [0u8; 32]);
        assert_eq!(state.next_nonce, 1);
    }

    #[test]
    fn state_survives_reload_and_rejects_replay() {
        let (db, path) = temp_db("replay");
        let supervisor = SupervisorKey::generate();

        // Turn 1: pin the supervisor, execute a manifest, persist.
        let mut state = load_kernel_state(&db, "agent-y");
        state.pinned_keys.insert(supervisor.public_pin());
        let mut manifest = Manifest {
            agent_id: "agent-y".into(),
            target_effect: "landscape".into(),
            arguments: br#"{"expr":"sin(x)"}"#.to_vec(),
            nonce: state.alloc_nonce(),
        };
        let grant = signed_grant(&manifest, &supervisor);
        state
            .consume_manifest_use_core(manifest.clone(), grant)
            .expect("first execution");
        save_kernel_state(&db, "agent-y", &state).expect("save");

        // Restart: new Database handle, new in-memory KernelState.
        let db2 = Database::open(&path).unwrap();
        let mut reloaded = load_kernel_state(&db2, "agent-y");
        assert_eq!(reloaded.next_nonce, 2, "nonce counter persisted");
        assert!(
            reloaded.pinned_keys.contains(&supervisor.public_pin()),
            "supervisor pin persisted"
        );
        assert!(
            reloaded.executed_nonces.contains(&manifest.nonce),
            "consumed nonce persisted"
        );
        assert_ne!(reloaded.current_history_root, [0u8; 32]);

        // The exact same grant must now be rejected (anti-replay across restart).
        let grant2 = signed_grant(&manifest, &supervisor);
        assert!(reloaded
            .consume_manifest_use_core(manifest.clone(), grant2)
            .is_err());

        // A fresh manifest with the persisted counter's nonce executes fine.
        manifest.nonce = reloaded.alloc_nonce();
        let grant3 = signed_grant(&manifest, &supervisor);
        assert!(reloaded
            .consume_manifest_use_core(manifest, grant3)
            .is_ok());
    }
}
