pub mod aukora;
pub mod config;
pub mod history;
pub mod interface;
pub mod orchestrator;
pub mod tools;

use llung_core::{
    identity::being::{Being, BeingKind},
    identity::machine::MachineIdentity,
    storage::db::Database,
};

pub struct AgentIdentity {
    pub machine: MachineIdentity,
    pub being: Being,
}

pub fn provision_agent(
    db: &Database,
    name: &str,
) -> Result<AgentIdentity, Box<dyn std::error::Error>> {
    let machine = db.create_machine(name)?;
    let being =
        db.create_being_for_machine(&machine.peer_id.to_base58(), name, BeingKind::Agent)?;

    Ok(AgentIdentity { machine, being })
}
