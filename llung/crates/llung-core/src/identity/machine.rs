use libp2p::{PeerId, identity::Keypair};

/// Ephemeral. Tied to one machine / one install.
#[derive(Clone, Debug)]
pub struct MachineIdentity {
    pub keypair: Keypair,
    pub peer_id: PeerId,
    pub device_name: String,
    pub being_id: String, // links back to Being
}
