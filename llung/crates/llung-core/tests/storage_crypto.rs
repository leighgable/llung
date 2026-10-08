use llung_core::identity::being::BeingKind;
use llung_core::identity::crypto::{decrypt_with_secret_key, encrypt_to_public_key};
use llung_core::network::message::{ChatMessage, MessageKind};
use llung_core::storage::db::Database;
use rand::rngs::SysRng;
use rand_core::UnwrapErr;
use x25519_dalek::{PublicKey, StaticSecret};

fn temp_db(tag: &str) -> String {
    let path = std::env::temp_dir().join(format!(
        "llung-test-{}-{}.db",
        tag,
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    path.to_string_lossy().into_owned()
}

#[test]
fn migrations_and_create_being_work() {
    let path = temp_db("create-being");
    let (db, _machine) = Database::new(&path).unwrap();

    let being = db.create_being("Alice", BeingKind::Human, None).unwrap();
    assert_eq!(being.human_name, "Alice");

    let loaded = db.load_being().unwrap().expect("being should persist");
    assert_eq!(loaded.being_id, being.being_id);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn create_being_for_machine_satisfies_enc_key_constraints() {
    let path = temp_db("create-being-for-machine");
    let (db, machine) = Database::new(&path).unwrap();

    let being = db
        .create_being_for_machine(&machine.peer_id.to_base58(), "Bot", BeingKind::Agent)
        .unwrap();
    assert_eq!(being.kind, BeingKind::Agent);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn message_kind_roundtrips_through_storage() {
    let path = temp_db("message-kind");
    let (db, _machine) = Database::new(&path).unwrap();

    let mut msg = ChatMessage {
        id: "m1".to_string(),
        topic: "introductions".to_string(),
        sender_id: "peer".to_string(),
        sender_name: "Alice".to_string(),
        parent_id: None,
        kind: MessageKind::ToolCall {
            tool_name: "weather".to_string(),
            arguments: r#"{"city":"Oslo"}"#.to_string(),
        },
        content: "what's the weather?".to_string(),
        timestamp: 42,
    };
    msg.id = msg.compute_id();

    db.save_message(&msg).unwrap();

    let loaded = db.load_messages_for_topic("introductions").unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].id, msg.id);
    match &loaded[0].kind {
        MessageKind::ToolCall { tool_name, arguments } => {
            assert_eq!(tool_name, "weather");
            assert_eq!(arguments, r#"{"city":"Oslo"}"#);
        }
        other => panic!("expected ToolCall, got {other:?}"),
    }

    // Non-struct variants survive too.
    let human = ChatMessage {
        id: "m2".to_string(),
        kind: MessageKind::Human,
        ..msg.clone()
    };
    db.save_message(&human).unwrap();
    let loaded = db.load_messages_for_topic("introductions").unwrap();
    assert_eq!(loaded[1].kind, MessageKind::Human);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn chacha_box_roundtrip_via_wrapper() {
    let mut rng = UnwrapErr(SysRng);
    let secret = StaticSecret::random_from_rng(&mut rng);
    let public = PublicKey::from(&secret);

    let ciphertext = encrypt_to_public_key(public.as_bytes(), b"secret invite").unwrap();
    let plaintext = decrypt_with_secret_key(secret.as_bytes(), &ciphertext).unwrap();

    assert_eq!(plaintext, b"secret invite");
}
