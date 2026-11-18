// llung-core/src/storage/dag.rs
//
// Inspired by Victor Taelin's OptChat (https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449).
//
// OptChat's "log" is append-only with permanent INTEGER indices — which
// presumes a single writer. Llung is multi-writer P2P, so no peer can
// assign i = 0,1,2,... without consensus. This module is the distributed
// generalization of that log:
//
//   * Every message extends exactly one parent, forming a DAG.
//   * A message's hash is H(parent_hash, content) — content-addressed,
//     so it needs no central allocator. No wall-clock timestamps or
//     senders in the hash: identity and ordering must not depend on
//     unverifiable metadata. (Sender identity still rides along in the
//     message and is gossipsub-signed.)
//   * The linear order is a pure function of the DAG: a deterministic
//     topological sort tie-broken by hash — the integer index replaced
//     by "the unique order every peer computes". Same DAG ⇒ same order,
//     no consensus (OptChat's core property, preserved).
//   * Messages whose parent we have not seen yet are buffered, not
//     dropped, and attached when the parent arrives. The parent's hash
//     is what a sync protocol would request next.
//   * epoch_root folds the *frontier* (childless tips — the newest end)
//     deterministically, so peers can compare "same chat?" in O(1).
//
// OptChat's OTHER components — the binary summary tree, the background
// compactor, and the budgeted view — live in crates/agent/src/history.rs.

use sha3::{Digest, Sha3_256};
use std::collections::{BTreeSet, HashMap, HashSet};

use crate::network::message::{ChatMessage, MessageKind};

#[derive(Debug, Clone)]
pub struct DagNode {
    pub message: ChatMessage,
    pub children: Vec<String>,
    /// sha3-256 of: parent_hash || content
    pub hash: [u8; 32],
}

/// ONE global DAG for all messages across all topics.
/// Topics are views (filters), not containers.
#[derive(Debug, Default)]
pub struct ChatDag {
    /// Root messages (no parent) across all topics
    pub roots: Vec<String>,
    /// All nodes by message id
    pub nodes: HashMap<String, DagNode>,
    /// Index: topic -> root ids for fast topic loading
    pub topic_roots: HashMap<String, Vec<String>>,
    /// Children whose parent has not arrived yet: parent id -> child ids.
    /// These are the hashes a sync protocol needs to request.
    pub pending: HashMap<String, Vec<String>>,
}

impl ChatDag {
    pub fn new() -> Self {
        Self::default()
    }

    /// H(parent_hash || content). Roots extend the zero hash.
    pub fn compute_hash(parent_hash: [u8; 32], content: &str) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(parent_hash);
        hasher.update(content.as_bytes());
        hasher.finalize().into()
    }

    /// Insert a message and compute its hash.
    /// Parents may arrive in any order — children are buffered until then.
    pub fn insert(&mut self, msg: ChatMessage) -> [u8; 32] {
        let id = msg.id.clone();
        let parent_id = msg.parent_id.clone();

        let parent_hash = parent_id
            .as_ref()
            .and_then(|pid| self.nodes.get(pid))
            .map(|parent| parent.hash)
            .unwrap_or([0u8; 32]);
        let hash = Self::compute_hash(parent_hash, &msg.content);

        match parent_id {
            Some(pid) if self.nodes.contains_key(&pid) => {
                self.nodes
                    .get_mut(&pid)
                    .expect("checked above")
                    .children
                    .push(id.clone());
            }
            Some(pid) => {
                // Parent not seen yet — buffer the child instead of
                // dropping it (it is not a root, so it would be unreachable).
                self.pending.entry(pid).or_default().push(id.clone());
            }
            None => {
                self.roots.push(id.clone());
                self.topic_roots
                    .entry(msg.topic.clone())
                    .or_default()
                    .push(id.clone());
            }
        }

        self.nodes.insert(
            id.clone(),
            DagNode {
                message: msg,
                children: Vec::new(),
                hash,
            },
        );

        // A newly arrived node may be the parent other nodes were waiting for.
        self.attach_pending(&id);

        hash
    }

    /// Link buffered children (recursively) to their freshly arrived parent.
    fn attach_pending(&mut self, parent_id: &str) {
        let Some(children) = self.pending.remove(parent_id) else {
            return;
        };
        if let Some(parent) = self.nodes.get_mut(parent_id) {
            parent.children.extend(children.iter().cloned());
        }
        for child in children {
            self.attach_pending(&child);
        }
    }

    /// Build a DAG from a flat list (e.g. from SQLite). Arrival order is
    /// irrelevant: linking is by id, and the canonical order is computed
    /// from structure at linearization time.
    pub fn build_from_flat_list(messages: Vec<ChatMessage>) -> Self {
        let mut dag = Self::new();
        for msg in messages {
            dag.insert(msg);
        }
        dag
    }

    /// Deterministic linearization of one topic (OptChat's "optimal" order):
    /// a topological sort that always emits the smallest-hash ready node.
    /// Pure function of the DAG — every peer converges to the same order.
    pub fn topic_linear(&self, topic: &str) -> Vec<&ChatMessage> {
        let topic_ids: HashSet<&String> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.message.topic == topic)
            .map(|(id, _)| id)
            .collect();

        // Ready set ordered by (node hash, id): the deterministic tie-break.
        let mut ready: BTreeSet<([u8; 32], &String)> = BTreeSet::new();
        let mut out: Vec<&ChatMessage> = Vec::with_capacity(topic_ids.len());

        // Nodes with no parent *inside this topic* are the starting frontier.
        for id in &topic_ids {
            let has_parent_in_topic = self.nodes[*id]
                .message
                .parent_id
                .as_ref()
                .is_some_and(|p| topic_ids.contains(p));
            if !has_parent_in_topic {
                ready.insert((self.nodes[*id].hash, *id));
            }
        }

        // Every message has exactly one parent, so a child becomes ready
        // the moment its parent is emitted.
        while let Some((_, id)) = ready.pop_first() {
            let node = &self.nodes[id];
            out.push(&node.message);
            for child in &node.children {
                if topic_ids.contains(child) {
                    ready.insert((self.nodes[child].hash, child));
                }
            }
        }

        out
    }

    /// The linear view of ONE branch: the ancestor chain of `id`, root
    /// first. Every node has exactly one parent, so this is always a
    /// plain chain — the "current branch" of the chat UI and of an
    /// agent's context. Unknown id yields an empty chain.
    pub fn ancestors_of(&self, id: &str) -> Vec<&ChatMessage> {
        let mut chain = Vec::new();
        let mut cur = self.nodes.get(id);
        while let Some(node) = cur {
            chain.push(&node.message);
            cur = node
                .message
                .parent_id
                .as_ref()
                .and_then(|pid| self.nodes.get(pid));
        }
        chain.reverse();
        chain
    }

    /// Childless tips of one topic — each is the head of a branch.
    pub fn leaves_of(&self, topic: &str) -> Vec<&String> {
        self.nodes
            .iter()
            .filter(|(_, node)| {
                node.message.topic == topic && node.children.is_empty()
            })
            .map(|(id, _)| id)
            .collect()
    }

    /// The tip an unfocused view follows: the leaf with the greatest
    /// (hash, id). Deterministic on purpose — the same choice every peer
    /// computes, independent of wall clocks (see the module header).
    pub fn tip_of(&self, topic: &str) -> Option<String> {
        self.leaves_of(topic)
            .into_iter()
            .max_by_key(|id| (self.nodes[*id].hash, (*id).clone()))
            .cloned()
    }

    /// Sibling branches at `id`'s fork point: the other children of its
    /// parent (or the other roots, for a root). The view cursor walks
    /// this list with next/prev.
    pub fn siblings_of(&self, id: &str) -> Vec<String> {
        let Some(node) = self.nodes.get(id) else {
            return Vec::new();
        };
        let mut sibs = match &node.message.parent_id {
            Some(pid) => self
                .nodes
                .get(pid)
                .map(|p| p.children.clone())
                .unwrap_or_default(),
            None => self
                .topic_roots
                .get(&node.message.topic)
                .cloned()
                .unwrap_or_default(),
        };
        sibs.retain(|s| s != id);
        sibs
    }

    /// The context view for an agent addressed on `trigger_id` (and the
    /// TUI's linear view of a branch): the trigger's ancestor chain, with
    /// every *non-human* reply hanging off that chain folded in — parallel
    /// agent replies, tool calls and their responses — ordered by
    /// (timestamp, hash).
    ///
    /// Human children that are not the chain continuation are explicit
    /// `/branch` forks: their whole subtree stays OUT of the context —
    /// unless the trigger itself lives on that fork, in which case the
    /// fork IS the chain (its own human children still isolate further
    /// sub-forks).
    ///
    /// Net effect: agents @'d on the same message all see each other's
    /// replies (the chat stays linear); only explicit forks isolate
    /// context. Views only — the canonical `topic_linear` is untouched.
    pub fn context_chain(&self, trigger_id: &str) -> Vec<&ChatMessage> {
        let chain = self.ancestors_of(trigger_id);
        let mut out = Vec::with_capacity(chain.len());
        for (i, msg) in chain.iter().enumerate() {
            out.push(*msg);
            let chain_child = chain.get(i + 1).map(|m| m.id.as_str());
            let Some(node) = self.nodes.get(&msg.id) else {
                continue;
            };
            let mut chatter: Vec<&String> = node
                .children
                .iter()
                .filter(|c| Some(c.as_str()) != chain_child)
                .filter(|c| {
                    !matches!(
                        self.nodes[*c].message.kind,
                        MessageKind::Human | MessageKind::Media(_)
                    )
                })
                .collect();
            chatter.sort_by_key(|c| {
                (
                    self.nodes[*c].message.timestamp,
                    self.nodes[*c].hash,
                    (*c).clone(),
                )
            });
            for c in chatter {
                self.collect_chatter(c, &mut out);
            }
        }
        out
    }

    /// Depth-first fold of a chatter subtree (agent reply trees,
    /// tool-call → response chains): everything under it is
    /// conversational by construction — include it all, deterministically
    /// ordered by (timestamp, hash).
    fn collect_chatter<'a>(&'a self, id: &str, out: &mut Vec<&'a ChatMessage>) {
        let Some(node) = self.nodes.get(id) else {
            return;
        };
        out.push(&node.message);
        let mut children: Vec<&String> = node.children.iter().collect();
        children.sort_by_key(|c| {
            (
                self.nodes[*c].message.timestamp,
                self.nodes[*c].hash,
                (*c).clone(),
            )
        });
        for c in children {
            self.collect_chatter(c, out);
        }
    }

    /// Ordered sibling branches at `id`'s fork point, INCLUDING `id`:
    /// the parent's children in this topic (or the topic's roots),
    /// ordered deterministically by (hash, id) — the list `/branch
    /// next`/`prev` walks.
    pub fn branch_options(&self, id: &str) -> Vec<String> {
        let Some(node) = self.nodes.get(id) else {
            return Vec::new();
        };
        let topic = node.message.topic.clone();
        let mut opts = match &node.message.parent_id {
            Some(pid) => self
                .nodes
                .get(pid)
                .map(|p| p.children.clone())
                .unwrap_or_default(),
            None => self.topic_roots.get(&topic).cloned().unwrap_or_default(),
        };
        opts.retain(|c| self.nodes[c].message.topic == topic);
        opts.sort_by_key(|c| (self.nodes[c].hash, c.clone()));
        opts
    }

    /// Global epoch root: a merkle fold over the sorted frontier (childless
    /// tips) — the newest end of the conversation. Sorting makes it a pure
    /// function of the DAG, so equal DAGs always yield equal roots and
    /// peers can detect divergence with one hash comparison.
    pub fn epoch_root(&self) -> [u8; 32] {
        let mut leaves: Vec<[u8; 32]> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.children.is_empty())
            .map(|(_, node)| node.hash)
            .collect();

        if leaves.is_empty() {
            return [0u8; 32];
        }
        if leaves.len() == 1 {
            return leaves[0];
        }

        leaves.sort_unstable();

        // Pad to power of 2
        let n = leaves.len().next_power_of_two();
        leaves.resize(n, [0u8; 32]);

        let mut width = n;
        while width > 1 {
            for i in 0..width / 2 {
                let mut h = Sha3_256::new();
                h.update(leaves[i * 2]);
                h.update(leaves[i * 2 + 1]);
                leaves[i] = h.finalize().into();
            }
            width /= 2;
        }
        leaves[0]
    }

    /// Self-certification check: recompute every node's hash from its
    /// parent's hash + content and report ids that don't match. On receipt
    /// of a message from the network this must be empty.
    pub fn check_integrity(&self) -> Vec<String> {
        let mut bad = Vec::new();
        // Walk in a stable order so the report is deterministic.
        let mut ids: Vec<&String> = self.nodes.keys().collect();
        ids.sort();
        for id in ids {
            let node = &self.nodes[id];
            let parent_hash = node
                .message
                .parent_id
                .as_ref()
                .and_then(|pid| self.nodes.get(pid))
                .map(|parent| parent.hash)
                .unwrap_or([0u8; 32]);
            if Self::compute_hash(parent_hash, &node.message.content) != node.hash {
                bad.push(id.clone());
            }
        }
        bad
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::message::MessageKind;

    fn msg(id: &str, topic: &str, parent: Option<&str>, content: &str) -> ChatMessage {
        ChatMessage {
            id: id.to_string(),
            topic: topic.to_string(),
            sender_id: "peer".to_string(),
            sender_name: "peer".to_string(),
            parent_id: parent.map(|p| p.to_string()),
            kind: MessageKind::Human,
            content: content.to_string(),
            timestamp: 0,
        }
    }

    #[test]
    fn order_is_deterministic_regardless_of_arrival_order() {
        // Two peers receive the same messages in different orders
        // (and with different local clocks) — the linearization
        // must be identical (OptChat's no-consensus convergence).
        let a = msg("a", "t", None, "hello");
        let b = msg("b", "t", Some("a"), "hi there");
        let c = msg("c", "t", Some("a"), "how are you?");

        let mut dag1 = ChatDag::new();
        dag1.insert(a.clone());
        dag1.insert(b.clone());
        dag1.insert(c.clone());

        let mut dag2 = ChatDag::new();
        dag2.insert(c);
        dag2.insert(a);
        dag2.insert(b);

        let order1: Vec<&str> = dag1.topic_linear("t").iter().map(|m| m.id.as_str()).collect();
        let order2: Vec<&str> = dag2.topic_linear("t").iter().map(|m| m.id.as_str()).collect();
        assert_eq!(order1, order2);
        assert_eq!(order1.len(), 3);
        assert_eq!(order1[0], "a"); // parent always precedes children
    }

    #[test]
    fn order_does_not_depend_on_timestamps() {
        // Same DAG, different wall-clock metadata: order must not change.
        let mut m1 = msg("x", "t", None, "first");
        let mut m2 = msg("y", "t", Some("x"), "second");
        m1.timestamp = 999;
        m2.timestamp = 1; // child claims to be OLDER than its parent

        let mut dag = ChatDag::new();
        dag.insert(m1);
        dag.insert(m2);

        let order: Vec<&str> = dag.topic_linear("t").iter().map(|m| m.id.as_str()).collect();
        assert_eq!(order, vec!["x", "y"]);
    }

    #[test]
    fn orphan_is_buffered_and_reattached_when_parent_arrives() {
        let mut dag = ChatDag::new();
        dag.insert(msg("child", "t", Some("parent"), "arrived early"));

        // It is buffered as a missing parent — NOT promoted to a root:
        assert!(dag.pending.contains_key("parent"));
        assert!(!dag.roots.iter().any(|r| r == "child"));

        // Until the parent arrives it shows as a detached frontier
        // (chat UX: never hide a message), still ordered by hash.
        let early: Vec<&str> = dag.topic_linear("t").iter().map(|m| m.id.as_str()).collect();
        assert_eq!(early, vec!["child"]);

        dag.insert(msg("parent", "t", None, "the root"));
        assert!(!dag.pending.contains_key("parent"));
        let order: Vec<&str> = dag.topic_linear("t").iter().map(|m| m.id.as_str()).collect();
        assert_eq!(order, vec!["parent", "child"]);
    }

    #[test]
    fn epoch_root_is_deterministic() {
        let mk = || {
            let mut dag = ChatDag::new();
            dag.insert(msg("a", "t", None, "one"));
            dag.insert(msg("b", "t", Some("a"), "two"));
            dag.insert(msg("c", "u", None, "other topic"));
            dag
        };
        assert_eq!(mk().epoch_root(), mk().epoch_root());

        // And it changes when the frontier changes.
        let mut dag = mk();
        dag.insert(msg("d", "t", Some("b"), "new tip"));
        assert_ne!(dag.epoch_root(), mk().epoch_root());
    }

    #[test]
    fn ancestors_form_the_branch_view() {
        let mut dag = ChatDag::new();
        dag.insert(msg("a", "t", None, "root"));
        dag.insert(msg("b", "t", Some("a"), "reply"));
        dag.insert(msg("c", "t", Some("b"), "fork one"));
        dag.insert(msg("d", "t", Some("b"), "fork two"));
        dag.insert(msg("e", "other", None, "different topic"));

        let chain: Vec<&str> = dag.ancestors_of("d").iter().map(|m| m.id.as_str()).collect();
        assert_eq!(chain, vec!["a", "b", "d"], "forks stay on their own branch");
        assert!(dag.ancestors_of("missing").is_empty());
        // The other topic never leaks into a branch view.
        assert!(!chain.contains(&"e"));
    }

    #[test]
    fn tip_is_the_greatest_leaf_and_siblings_share_a_parent() {
        let mut dag = ChatDag::new();
        dag.insert(msg("a", "t", None, "root"));
        dag.insert(msg("b", "t", Some("a"), "main"));
        dag.insert(msg("c", "t", Some("a"), "side branch"));
        dag.insert(msg("d", "u", None, "other topic"));

        let tip = dag.tip_of("t").expect("a tip");
        assert!(dag.nodes[&tip].children.is_empty());
        assert_eq!(dag.nodes[&tip].message.topic, "t");
        assert_ne!(tip_of_view(&dag, "t"), "d", "other topic's leaf excluded");

        let mut sibs = dag.siblings_of("b");
        sibs.sort();
        assert_eq!(sibs, vec!["c".to_string()]);
        // Roots are siblings of each other.
        let mut root_sibs = dag.siblings_of("a");
        root_sibs.sort();
        assert!(root_sibs.is_empty(), "only root in topic t");
        assert!(dag.siblings_of("missing").is_empty());
    }

    fn tip_of_view(dag: &ChatDag, topic: &str) -> String {
        dag.tip_of(topic).unwrap()
    }

    #[test]
    fn context_chain_keeps_parallel_agents_linear_and_forks_out() {
        let agent_msg = |id: &str, topic: &str, parent: Option<&str>, content: &str, ts: u64| {
            let mut m = msg(id, topic, parent, content);
            m.kind = MessageKind::Agent;
            m.timestamp = ts;
            m
        };
        let mut dag = ChatDag::new();
        dag.insert(msg("h1", "t", None, "hello"));
        dag.insert(msg("h2", "t", Some("h1"), "agents, compare notes"));
        // Two agents @'d on h2 in parallel, plus a tool call + response.
        dag.insert(agent_msg("a1", "t", Some("h2"), "from agent one", 30));
        dag.insert(agent_msg("t1", "t", Some("h2"), "TOOL {}", 31));
        dag.insert(agent_msg("r1", "t", Some("t1"), "tool output", 32));
        dag.insert(agent_msg("a2", "t", Some("h2"), "from agent two", 33));
        // An explicit /branch fork off h1, with an agent reply on it.
        let mut f1 = msg("f1", "t", Some("h1"), "forked aside");
        f1.timestamp = 20;
        dag.insert(f1);
        dag.insert(agent_msg("fa", "t", Some("f1"), "agent on the fork", 21));

        // An agent addressed on the main chain sees the OTHER agent's
        // reply and the tool traffic — the conversation is linear.
        let ctx: Vec<&str> = dag
            .context_chain("h2")
            .iter()
            .map(|m| m.id.as_str())
            .collect();
        assert_eq!(ctx, vec!["h1", "h2", "a1", "t1", "r1", "a2"]);
        assert!(!ctx.contains(&"f1"), "explicit fork excluded");
        assert!(!ctx.contains(&"fa"));

        // An agent addressed on the fork sees the fork chain — and the
        // fork's own chatter, but not the main branch's humans.
        let fctx: Vec<&str> = dag
            .context_chain("fa")
            .iter()
            .map(|m| m.id.as_str())
            .collect();
        assert_eq!(fctx, vec!["h1", "f1", "fa"]);
    }

    #[test]
    fn hash_chains_are_self_certifying() {
        let mut dag = ChatDag::new();
        dag.insert(msg("a", "t", None, "root"));
        dag.insert(msg("b", "t", Some("a"), "child"));
        assert!(dag.check_integrity().is_empty());

        // Tamper with content in place: detected.
        if let Some(node) = dag.nodes.get_mut("b") {
            node.message.content = "tampered".to_string();
        }
        assert_eq!(dag.check_integrity(), vec!["b".to_string()]);
    }
}
