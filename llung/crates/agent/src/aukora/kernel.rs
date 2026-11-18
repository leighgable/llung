use bincode::Options;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Manifest {
    pub agent_id: String,
    pub target_effect: String, // e.g., "db:write" or "network:send"
    pub arguments: Vec<u8>,    // Arbitrary payload requested by the agent
    pub nonce: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Grant {
    pub manifest_hash: [u8; 32],
    pub authorized_by: [u8; 32], // Public key pin of the supervisor/operator
    pub signature: Vec<u8>,      // Cryptographic signature over the hash
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Receipt {
    pub manifest_hash: [u8; 32],
    pub status: ReceiptStatus,
    pub output_commitment: [u8; 32], // Hash of the action's result
}

impl Receipt {
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::options()
            .with_little_endian()
            .with_fixint_encoding()
            .serialize(self)
            .map_err(|e| format!("Deterministic serialization failed: {}", e))
    }
    pub fn compute_hash(&self) -> Result<[u8; 32], String> {
        let bytes = self.to_bytes()?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        Ok(hasher.finalize().into())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptStatus {
    Executed,
    Denied,
    Replayed,
}

/// A standing (perpetual) grant: authorizes every manifest matching
/// (agent_id, effect), argument-agnostic, without consuming nonces.
/// Created by the operator via `/grant <effect> perpetual` or the
/// prompt's "always" answer; `authorized_by` is the supervisor pin that
/// vouched for it.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct StandingGrant {
    pub agent_id: String,
    pub effect: String,
    pub authorized_by: [u8; 32],
}

pub struct KernelState {
    pub pinned_keys: HashSet<[u8; 32]>, // Strict key pinning vs TOFU
    pub executed_nonces: HashSet<u64>,  // Anti-replay state tracking
    pub current_history_root: [u8; 32], // Current Merkle root
    /// Monotonic allocation counter for manifest nonces. Persisted so a
    /// restart can never re-issue a nonce that was already consumed.
    pub next_nonce: u64,
    /// Operator-approved perpetual grants (see `StandingGrant`).
    pub standing_grants: Vec<StandingGrant>,
}

impl Default for KernelState {
    fn default() -> Self {
        Self {
            pinned_keys: HashSet::new(),
            executed_nonces: HashSet::new(),
            current_history_root: [0u8; 32],
            next_nonce: 1,
            standing_grants: Vec::new(),
        }
    }
}

impl KernelState {
    /// Allocate the next manifest nonce. Gaps are fine (crash between
    /// allocation and execution); reuse is what anti-replay forbids, so
    /// the counter only ever moves forward and is persisted.
    pub fn alloc_nonce(&mut self) -> u64 {
        let nonce = self.next_nonce;
        self.next_nonce += 1;
        nonce
    }

    /// True when a standing grant covers this manifest. Standing grants
    /// skip the grant-signature and nonce-consumption path entirely —
    /// the operator pre-authorized the (agent, effect) pair.
    pub fn check_standing(&self, manifest: &Manifest) -> bool {
        self.standing_grants.iter().any(|g| {
            g.agent_id == manifest.agent_id && g.effect == manifest.target_effect
        })
    }

    /// Add a standing grant; false when an identical rule exists.
    pub fn grant_standing(&mut self, grant: StandingGrant) -> bool {
        if self.check_standing(&Manifest {
            agent_id: grant.agent_id.clone(),
            target_effect: grant.effect.clone(),
            arguments: Vec::new(),
            nonce: 0,
        }) {
            return false;
        }
        self.standing_grants.push(grant);
        true
    }

    /// Remove a standing grant; true when one was removed.
    pub fn revoke_standing(&mut self, agent_id: &str, effect: &str) -> bool {
        let before = self.standing_grants.len();
        self.standing_grants
            .retain(|g| !(g.agent_id == agent_id && g.effect == effect));
        self.standing_grants.len() != before
    }

    /// Record a DENIED action in the append-only history: the receipt
    /// folds into the merkle root without a grant being consumed, so a
    /// retry (with a fresh ask) stays possible and the denial is auditable.
    pub fn record_denial(&mut self, manifest: &Manifest) -> Receipt {
        let receipt = Receipt {
            manifest_hash: compute_sha256(manifest),
            status: ReceiptStatus::Denied,
            output_commitment: [0u8; 32],
        };
        self.advance_history_root(&receipt);
        receipt
    }
}

/// Canonical manifest/receipt hashing. Grant verification recomputes this,
/// so there must be exactly ONE code path (same serialization, same hash).
pub fn compute_sha256<T: Serialize>(value: &T) -> [u8; 32] {
    let bytes = bincode::options()
        .with_little_endian()
        .with_fixint_encoding()
        .serialize(value)
        .expect("kernel structs are trivially serializable");
    compute_sha256_bytes(&bytes)
}

/// Hash of raw bytes (e.g. a tool's output commitment).
pub fn compute_sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Signature verification against the pinned supervisor key. Real
/// ed25519: an empty or malformed signature is rejected outright (the
/// old stub accepted it, which made self-grants unverifiable in spirit).
pub fn verify_signature(signature: &[u8], hash: &[u8; 32], key: &[u8; 32]) -> bool {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};
    let Ok(key) = VerifyingKey::from_bytes(key) else {
        return false;
    };
    let Ok(signature) = Signature::from_slice(signature) else {
        return false;
    };
    key.verify(hash, &signature).is_ok()
}

/// Append-only history: fold the new leaf into the running merkle root.
pub fn update_merkle_root(current: [u8; 32], leaf: [u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(current);
    hasher.update(leaf);
    hasher.finalize().into()
}

impl KernelState {
    pub fn consume_manifest_use_core(
        &mut self,
        manifest: Manifest,
        grant: Grant,
    ) -> Result<Receipt, &'static str> {
        // 1. Strict anti-replay check
        if self.executed_nonces.contains(&manifest.nonce) {
            return Err("Replay attack detected: Nonce already used.");
        }

        // 2. Validate authorizer signature is actually on the pinned list
        if !self.pinned_keys.contains(&grant.authorized_by) {
            return Err("Security Violation: Authorization key not explicitly pinned.");
        }

        // 3. Compute manifest cryptographic hash (e.g., via SHA-256 or a post-quantum library)
        let computed_hash = compute_sha256(&manifest);
        if computed_hash != grant.manifest_hash {
            return Err("Integrity Error: Grant hash does not match proposed Manifest.");
        }

        // 4. Verify signature (Using a crate like `ed25519-dalek` or similar)
        if !verify_signature(&grant.signature, &computed_hash, &grant.authorized_by) {
            return Err("Invalid Signature: Grant authorization failed.");
        }

        // 5. Update State
        self.executed_nonces.insert(manifest.nonce);

        // 6. Append to Merkle History Tree
        self.current_history_root = update_merkle_root(self.current_history_root, computed_hash);

        Ok(Receipt {
            manifest_hash: computed_hash,
            status: ReceiptStatus::Executed,
            output_commitment: [0u8; 32], // Populated once the host layer runs the effect
        })
    }
    pub fn advance_history_root(&mut self, receipt: &Receipt) {
        let mut receipt_hasher = Sha256::new();
        receipt_hasher.update(&receipt.manifest_hash);
        receipt_hasher.update(&[receipt.status as u8]);
        receipt_hasher.update(&receipt.output_commitment);
        let receipt_hash = receipt_hasher.finalize();

        // 2. Combine the old history root with the new receipt hash (Chaining)
        let mut root_hasher = Sha256::new();
        root_hasher.update(&self.current_history_root);
        root_hasher.update(&receipt_hash);

        // 3. Commit the new root back into the kernel state
        self.current_history_root = root_hasher.finalize().into();
    }
}
