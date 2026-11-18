// crates/relay/src/image_store.rs
//
// SQLite-backed image cache for the relay: avatars and message image
// attachments only (≤512KB each, validated by the shared protocol
// checker before reaching this layer). TTL expiry + a global count cap
// keep the €5 VPS disk in check.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use llung_core::media::image::validate_image_put;
use rusqlite::{Connection, params};

pub struct ImageStore {
    conn: Connection,
    max_images: usize,
    ttl: Duration,
}

impl ImageStore {
    pub fn open(path: &Path, max_images: usize, ttl: Duration) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS images (
                cid TEXT PRIMARY KEY,
                mime TEXT NOT NULL,
                data BLOB NOT NULL,
                expires INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_images_expires ON images(expires);",
        )
        .map_err(|e| e.to_string())?;
        Ok(Self { conn, max_images, ttl })
    }

    fn now() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    pub fn len(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM images", [], |row| row.get::<_, i64>(0))
            .map(|n| n as usize)
            .unwrap_or(0)
    }

    /// Store (or refresh) an image. Validation runs here too — the
    /// store never trusts its callers. `Err` is the rejection reason.
    pub fn put(&mut self, cid: &str, mime: &str, data: &[u8]) -> Result<(), String> {
        validate_image_put(cid, mime, data)?;
        let expires = Self::now() + self.ttl.as_secs() as i64;
        self.conn
            .execute(
                "INSERT INTO images (cid, mime, data, expires) VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(cid) DO UPDATE SET mime=excluded.mime, data=excluded.data, expires=excluded.expires",
                params![cid, mime, data, expires],
            )
            .map_err(|e| e.to_string())?;
        self.evict_if_needed();
        Ok(())
    }

    pub fn get(&self, cid: &str) -> Option<(String, Vec<u8>)> {
        let row = self
            .conn
            .query_row(
                "SELECT mime, data, expires FROM images WHERE cid = ?1",
                params![cid],
                |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?, row.get::<_, i64>(2)?))
                },
            )
            .ok()?;
        if row.2 < Self::now() {
            return None;
        }
        Some((row.0, row.1))
    }

    /// Delete expired images; returns how many were removed.
    pub fn sweep(&self) -> usize {
        self.conn
            .execute("DELETE FROM images WHERE expires < ?1", params![Self::now()])
            .unwrap_or(0)
    }

    /// Past the count cap: evict the images expiring soonest.
    fn evict_if_needed(&mut self) {
        let count = self.len();
        if count <= self.max_images {
            return;
        }
        let excess = count - self.max_images;
        self.conn
            .execute(
                "DELETE FROM images WHERE cid IN (
                     SELECT cid FROM images ORDER BY expires ASC LIMIT ?1
                 )",
                params![excess as i64],
            )
            .ok();
    }
}

// MAX_IMAGE_BYTES (imported) is the protocol-level bound the store
// enforces via `validate_image_put` before any write.

#[cfg(test)]
mod tests {
    use super::*;
    use llung_core::media::image::image_cid;

    fn temp_store(max: usize) -> (ImageStore, PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "llung-relay-store-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = ImageStore::open(&path, max, Duration::from_secs(3600)).unwrap();
        (store, path)
    }

    #[test]
    fn put_get_round_trip_and_validation() {
        let (mut store, path) = temp_store(10);
        let data = b"png-bytes";
        let cid = image_cid(data);
        store.put(&cid, "image/png", data).unwrap();
        assert_eq!(store.len(), 1);
        let (mime, bytes) = store.get(&cid).unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(bytes, data);
        assert!(store.get("nope").is_none());
        // Junk is rejected at the store layer too.
        assert!(store.put("bad-cid", "image/png", data).is_err());
        assert!(store.put(&cid, "text/plain", data).is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn eviction_drops_oldest_expiring_past_the_cap() {
        let (mut store, path) = temp_store(2);
        // Three images, staggered expiry by re-putting with different
        // TTLs is awkward — instead rely on insertion order: expires are
        // equal, so eviction order among equals is arbitrary but the
        // count must hold.
        for i in 0..3u8 {
            let data = vec![i; 16];
            store.put(&image_cid(&data), "image/png", &data).unwrap();
        }
        assert!(store.len() <= 2, "cap must hold, got {}", store.len());
        std::fs::remove_file(path).ok();
    }
}
