use crate::aukora::kernel::{Receipt, ReceiptStatus};
use sha2::{Digest, Sha256};

pub struct PeerStateVerifier;

impl PeerStateVerifier {
    pub fn verify_state_transition(
        last_known_history_root: &[u8; 32],
        proposed_history_root: &[u8; 32],
        incoming_receipt: &Receipt,
    ) -> Result<(), String> {
        // nsure the receipt wasn't dynamically marked as a failure
        // if your security policy strictly forbids emitting corrupted logs.
        if incoming_receipt.status == ReceiptStatus::Replayed {
            return Err("Rejected: Local validation caught a replayed action.".into());
        }

        // hash the incoming receipt deterministically
        let receipt_hash = incoming_receipt.compute_hash()?;

        // roll the history root forward locally using the last known consensus state
        let mut root_hasher = Sha256::new();
        root_hasher.update(last_known_history_root);
        root_hasher.update(&receipt_hash);
        let locally_computed_root: [u8; 32] = root_hasher.finalize().into();

        // cryptographic validation
        if &locally_computed_root != proposed_history_root {
            return Err(
                "State Integrity Mismatch: The agent payload or state order has been altered!"
                    .into(),
            );
        }

        // if everything checks out the peer can safely apply the state transition to their local UI.
        Ok(())
    }
}
