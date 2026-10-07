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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ReceiptStatus {
    Executed,
    Denied,
    Replayed,
}

pub struct KernelState {
    pub pinned_keys: HashSet<[u8; 32]>, // Strict key pinning vs TOFU
    pub executed_nonces: HashSet<u64>,  // Anti-replay state tracking
    pub current_history_root: [u8; 32], // Current Merkle root
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
