// crates/agent/src/aukora/supervisor.rs
//
// The supervisor key: the local human operator's ed25519 identity for
// signing Aukora grants. Generated fresh when an agent is spawned (the
// "user who spawned this agent is the supervisor"), public half pinned
// in the kernel and persisted alongside the agent's kernel state.
//
// The signing key itself stays in memory — a restart mints a new
// supervisor key, which is fine: consumed nonces and the merkle chain
// are what persist, not past grants' verifiability against a dead key.

use ed25519_dalek::{Signer, SigningKey};
use rand::rand_core::Rng;

/// ed25519 keypair that authorizes manifest execution by signing the
/// manifest hash embedded in each `Grant`.
pub struct SupervisorKey {
    signing: SigningKey,
}

impl SupervisorKey {
    /// Generate a fresh supervisor identity from OS randomness.
    /// Registration persists the seed in the account DB; use
    /// [`from_seed`] everywhere except tests.
    pub fn generate() -> Self {
        let mut rng = rand::rand_core::UnwrapErr(rand::rngs::SysRng);
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        Self::from_seed(&bytes)
    }

    /// Rebuild the supervisor identity from its persisted seed (the
    /// account's `aukora_supervisor` row).
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        Self {
            signing: SigningKey::from_bytes(seed),
        }
    }

    /// The seed bytes for persistence.
    pub fn seed(&self) -> [u8; 32] {
        self.signing.to_bytes()
    }

    /// The 32-byte public key pin to insert into `KernelState::pinned_keys`.
    pub fn public_pin(&self) -> [u8; 32] {
        self.signing.verifying_key().to_bytes()
    }

    /// Sign a manifest hash, producing the grant's signature bytes.
    pub fn sign(&self, manifest_hash: &[u8; 32]) -> Vec<u8> {
        self.signing.sign(manifest_hash).to_bytes().to_vec()
    }
}

impl std::fmt::Debug for SupervisorKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupervisorKey")
            .field("public_pin", &self.public_pin())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aukora::kernel::{compute_sha256, verify_signature};

    #[test]
    fn generated_key_signs_and_verifies() {
        let key = SupervisorKey::generate();
        let hash = compute_sha256(&"some manifest");
        let sig = key.sign(&hash);
        assert!(verify_signature(&sig, &hash, &key.public_pin()));
    }

    #[test]
    fn wrong_key_and_tampered_hash_fail() {
        let key = SupervisorKey::generate();
        let other = SupervisorKey::generate();
        let hash = compute_sha256(&"some manifest");
        let sig = key.sign(&hash);
        // A grant signed by a different supervisor must not verify.
        assert!(!verify_signature(&sig, &hash, &other.public_pin()));
        // Nor a valid signature against a tampered manifest hash.
        let mut tampered = hash;
        tampered[0] ^= 1;
        assert!(!verify_signature(&sig, &tampered, &key.public_pin()));
    }

    #[test]
    fn empty_or_malformed_signatures_are_rejected() {
        let key = SupervisorKey::generate();
        let hash = compute_sha256(&"some manifest");
        assert!(!verify_signature(&[], &hash, &key.public_pin()));
        assert!(!verify_signature(&[1, 2, 3], &hash, &key.public_pin()));
        assert!(!verify_signature(&vec![0u8; 64], &hash, &key.public_pin()));
    }
}
