// llung-core/src/storage/dag.rs
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

use crate::network::message::ChatMessage;

#[derive(Debug, Clone)]
pub struct DagNode {
    pub message: ChatMessage,
    pub children: Vec<String>,
    /// sha3-256 of: content || sender_id || timestamp || parent_hashes
    pub hash: [u8; 32],
}

/// ONE global DAG for all messages across all topics.
/// Topics are views (filters), not containers.
#[derive(Debug, Default)]
pub struct ChatDag {
    /// All root messages (no parent) across all topics
    pub roots: Vec<String>,
    /// All nodes by message id
    pub nodes: HashMap<String, DagNode>,
    /// Index: topic -> root ids for fast topic loading
    pub topic_roots: HashMap<String, Vec<String>>,
}

impl ChatDag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a message and compute its hash.
    /// Parents must already exist (insert in chronological order).
    pub fn insert(&mut self, msg: ChatMessage) -> [u8; 32] {
        let id = msg.id.clone();
        let parent_id = msg.parent_id.clone();

        // Compute hash
        let mut hasher = Sha3_256::new();
        hasher.update(msg.content.as_bytes());
        hasher.update(msg.sender_id.as_bytes());
        hasher.update(msg.timestamp.to_le_bytes());
        if let Some(ref pid) = parent_id {
            if let Some(parent) = self.nodes.get(pid) {
                hasher.update(&parent.hash);
            }
        }
        let hash: [u8; 32] = hasher.finalize().into();

        // Link to parent
        if let Some(ref pid) = parent_id {
            if let Some(parent) = self.nodes.get_mut(pid) {
                parent.children.push(id.clone());
            }
        } else {
            self.roots.push(id.clone());
            self.topic_roots
                .entry(msg.topic.clone())
                .or_default()
                .push(id.clone());
        }

        self.nodes.insert(
            id,
            DagNode {
                message: msg,
                children: Vec::new(),
                hash,
            },
        );

        hash
    }

    pub fn build_from_flat_list(mut messages: Vec<ChatMessage>) -> Self {
        messages.sort_by_key(|m| m.timestamp);
        let mut dag = Self::new();
        for msg in messages {
            dag.insert(msg);
        }
        dag
    }

    /// Linearize one topic (chronological DFS).
    pub fn topic_linear(&self, topic: &str) -> Vec<&ChatMessage> {
        let mut out = Vec::new();
        let root_ids = self.topic_roots.get(topic).cloned().unwrap_or_default();
        let mut stack: Vec<String> = root_ids;

        while let Some(id) = stack.pop() {
            if let Some(node) = self.nodes.get(&id) {
                if node.message.topic == topic {
                    out.push(&node.message);
                }
                for child_id in node.children.iter().rev() {
                    stack.push(child_id.clone());
                }
            }
        }
        out
    }

    /// Global epoch root: hash of all current root hashes.
    /// Two peers compare this to detect divergence.
    pub fn epoch_root(&self) -> [u8; 32] {
        let mut leaves: Vec<[u8; 32]> = self
            .roots
            .iter()
            .filter_map(|id| self.nodes.get(id).map(|n| n.hash))
            .collect();

        if leaves.is_empty() {
            return [0u8; 32];
        }
        if leaves.len() == 1 {
            return leaves[0];
        }

        // Pad to power of 2
        let n = leaves.len().next_power_of_two();
        leaves.resize(n, [0u8; 32]);

        let mut width = n;
        while width > 1 {
            for i in 0..width / 2 {
                let mut h = Sha3_256::new();
                h.update(&leaves[i * 2]);
                h.update(&leaves[i * 2 + 1]);
                leaves[i] = h.finalize().into();
            }
            width /= 2;
        }
        leaves[0]
    }
}
