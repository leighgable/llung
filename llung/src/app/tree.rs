use std::collections::HashMap;

use crate::network::message::ChatMessage;

#[derive(Debug, Clone)]
pub struct ChatNode {
    pub message: ChatMessage,
    pub children: Vec<String>, // Stores the IDs of child messages
}

#[derive(Debug, Default)]
pub struct ChatTree {
    pub root_ids: Vec<String>, // Messages with no parent
    pub nodes: HashMap<String, ChatNode>,
}

impl ChatTree {
    /// Feed a flat list of messages from SQLite into this function
    /// to build the tree in memory.
    pub fn build_from_flat_list(messages: Vec<ChatMessage>) -> Self {
        let mut tree = ChatTree::default();

        for msg in messages {
            let id = msg.id.clone();
            let parent_id = msg.parent_id.clone();

            tree.nodes.insert(
                id.clone(),
                ChatNode {
                    message: msg,
                    children: Vec::new(),
                },
            );

            if let Some(parent) = parent_id {
                // If the parent is already in the tree, add this node to its children
                if let Some(parent_node) = tree.nodes.get_mut(&parent) {
                    parent_node.children.push(id);
                }
            } else {
                // No parent means it's a root message
                tree.root_ids.push(id);
            }
        }
        tree
    }
}
