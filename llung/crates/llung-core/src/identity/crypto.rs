use crypto_box::{
    ChaChaBox, Nonce, PublicKey, SecretKey,
    aead::{Aead, AeadCore},
};
use rand::{Rng, rngs::SysRng};
use rand_core::UnwrapErr;

// A simple compatibility bridge
pub struct LegacyRng<'a>(&'a mut UnwrapErr<SysRng>);

// Implement the legacy RngCore that crypto_box expects
impl<'a> crypto_box::aead::rand_core::RngCore for LegacyRng<'a> {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }
    fn try_fill_bytes(
        &mut self,
        dest: &mut [u8],
    ) -> Result<(), crypto_box::aead::rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}
// Implement the legacy marker trait crypto_box expects
impl<'a> crypto_box::aead::rand_core::CryptoRng for LegacyRng<'a> {}

/// Encrypt `plaintext` to a recipient's X25519 public key.
/// Returns: ephemeral_public_key (32 bytes) || nonce (24 bytes) || ciphertext
pub fn encrypt_to_public_key(
    recipient_public_key: &[u8; 32],
    plaintext: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut native_rng = UnwrapErr(SysRng);
    let mut rng = LegacyRng(&mut native_rng);
    let recipient = PublicKey::from(*recipient_public_key);

    // Ephemeral keypair for each message (forward secrecy)
    let ephemeral_secret = SecretKey::generate(&mut rng);
    let ephemeral_public = ephemeral_secret.public_key();

    let box_instance = ChaChaBox::new(&recipient, &ephemeral_secret);
    let nonce = ChaChaBox::generate_nonce(&mut rng);

    let ciphertext = box_instance
        .encrypt(&nonce, plaintext)
        .map_err(|e| format!("encryption failed: {:?}", e))?;

    // Serialize: ephemeral_pk || nonce || ciphertext
    let mut output = Vec::with_capacity(32 + 24 + ciphertext.len());
    output.extend_from_slice(ephemeral_public.as_bytes());
    output.extend_from_slice(nonce.as_ref());
    output.extend_from_slice(&ciphertext);

    Ok(output)
}

/// Decrypt a message using our X25519 secret key.
/// Input format: ephemeral_public_key (32) || nonce (24) || ciphertext
pub fn decrypt_with_secret_key(
    secret_key: &[u8; 32],
    ciphertext_box: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if ciphertext_box.len() < 32 + 24 {
        return Err("ciphertext too short".into());
    }

    let ephemeral_pk_bytes: [u8; 32] = ciphertext_box[..32].try_into().unwrap();
    let nonce_bytes: [u8; 24] = ciphertext_box[32..56].try_into().unwrap();
    let ciphertext = &ciphertext_box[56..];

    let secret = SecretKey::from(*secret_key);
    let ephemeral_public = PublicKey::from(ephemeral_pk_bytes);
    let nonce = Nonce::from(nonce_bytes);

    let box_instance = ChaChaBox::new(&ephemeral_public, &secret);

    box_instance
        .decrypt(&nonce, ciphertext)
        .map_err(|e| format!("decryption failed: {:?}", e).into())
}
