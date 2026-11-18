// llung-core/src/media/image.rs
//
// Lightweight image transfer: the `/llung/img/1.0.0` request-response
// protocol shared by every node and the relay. Scope is deliberately
// narrow — PEER AVATARS and MESSAGE IMAGE ATTACHMENTS:
//
// - Blobs are small (MAX_IMAGE_BYTES). Anything bigger travels by other
//   means later; the relay is a €5 VPS, not a CDN.
// - Puts are validated (size cap, sha256(cid) == data, image/* only) on
//   BOTH sides, so a junk-spewing peer can't fill the store.
// - `Put` is an upload to whoever serves it (relay cache, or a peer that
//   asked); `Get` fetches by content id.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// libp2p stream protocol for image transfer.
pub const IMAGE_PROTOCOL: &str = "/llung/img/1.0.0";
/// Hard cap per image (avatars and inline message images only).
pub const MAX_IMAGE_BYTES: usize = 512 * 1024;
/// identify agent_version a relay server advertises; clients use it to
/// pick whom to cache images on / fetch images from.
pub const RELAY_AGENT_VERSION: &str = "/llung/relay/1.0.0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageRequest {
    /// Upload a small image into the peer's/relay's cache.
    Put {
        cid: String,
        mime_type: String,
        data: Vec<u8>,
    },
    /// Fetch a cached image by content id.
    Get { cid: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageResponse {
    PutAck { accepted: bool, reason: Option<String> },
    Got { mime_type: String, data: Vec<u8> },
    NotFound,
}

/// sha256 hex digest — the same content id `generate_cid_from_bytes`
/// produces elsewhere in the codebase.
pub fn image_cid(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Validation every receiver applies before storing or acking a Put.
/// Returns the rejection reason, if invalid.
pub fn validate_image_put(cid: &str, mime_type: &str, data: &[u8]) -> Result<(), String> {
    if data.is_empty() {
        return Err("empty image".into());
    }
    if data.len() > MAX_IMAGE_BYTES {
        return Err(format!(
            "{} bytes exceeds the {}-byte lightweight cap",
            data.len(),
            MAX_IMAGE_BYTES
        ));
    }
    if !mime_type.starts_with("image/") {
        return Err(format!("not an image mime: {mime_type}"));
    }
    if image_cid(data) != cid {
        return Err("cid does not match the image data".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_good_puts() {
        let data = b"fake-png-bytes";
        let cid = image_cid(data);
        assert!(validate_image_put(&cid, "image/png", data).is_ok());
    }

    #[test]
    fn validate_rejects_junk() {
        let data = b"fake-png-bytes";
        let cid = image_cid(data);
        assert!(validate_image_put("deadbeef", "image/png", data).is_err());
        assert!(validate_image_put(&cid, "text/plain", data).is_err());
        assert!(validate_image_put(&cid, "image/png", &[]).is_err());
        let big = vec![0u8; MAX_IMAGE_BYTES + 1];
        assert!(validate_image_put(&image_cid(&big), "image/png", &big).is_err());
    }

    #[test]
    fn request_response_round_trip_through_cbor_shape() {
        let req = ImageRequest::Put {
            cid: "abc".into(),
            mime_type: "image/png".into(),
            data: vec![1, 2, 3],
        };
        let json = serde_json::to_value(&req).unwrap();
        let back: ImageRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req, back);
        let resp = ImageResponse::NotFound;
        let back: ImageResponse =
            serde_json::from_value(serde_json::to_value(&resp).unwrap()).unwrap();
        assert_eq!(resp, back);
    }
}
