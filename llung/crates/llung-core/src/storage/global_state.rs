use sha3::{Digest, Sha3_256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct GlobalState {
    topic_roots: BTreeMap<String, [u8; 32]>,
}

impl GlobalState {
    pub fn update_topic(&mut self, topic: String, root: [u8; 32]) {
        self.topic_roots.insert(topic, root);
    }

    pub fn global_root(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        for (topic, root) in &self.topic_roots {
            hasher.update(topic.as_bytes());
            hasher.update(root);
        }
        hasher.finalize().into()
    }

    pub fn diff_topics(&self, other: &GlobalState) -> Vec<String> {
        let mut divergent = Vec::new();
        for (topic, my_root) in &self.topic_roots {
            match other.topic_roots.get(topic) {
                Some(their_root) if my_root == their_root => {}
                _ => divergent.push(topic.clone()),
            }
        }
        for topic in other.topic_roots.keys() {
            if !self.topic_roots.contains_key(topic) {
                divergent.push(topic.clone());
            }
        }
        divergent
    }
}
