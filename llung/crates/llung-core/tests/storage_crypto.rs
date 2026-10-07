use llung_core::identity::being::BeingKind;
use llung_core::identity::crypto::{decrypt_with_secret_key, encrypt_to_public_key};
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
fn chacha_box_roundtrip_via_wrapper() {
    let mut rng = UnwrapErr(SysRng);
    let secret = StaticSecret::random_from_rng(&mut rng);
    let public = PublicKey::from(&secret);

    let ciphertext = encrypt_to_public_key(public.as_bytes(), b"secret invite").unwrap();
    let plaintext = decrypt_with_secret_key(secret.as_bytes(), &ciphertext).unwrap();

    assert_eq!(plaintext, b"secret invite");
}
