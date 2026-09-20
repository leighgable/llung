pub const CREATE_MESSAGES_TABLE: &str = r#"
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,      
        topic TEXT NOT NULL,
        parent_id TEXT,           
        sender_id TEXT NOT NULL,
        sender_name TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp INTEGER DEFAULT (unixepoch())
    );

CREATE INDEX IF NOT EXISTS idx_messages_topic ON messages(topic);
CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id);


"#;

pub const CREATE_AVATAR_CACHE_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS avatar_cache (
        cid TEXT PRIMARY KEY,
        image_bytes BLOB NOT NULL,
        created_at INTEGER NOT NULL
    );
";

pub const CREATE_BEINGS_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS beings (
        being_id TEXT PRIMARY KEY,
        human_name TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'human',
        status TEXT NOT NULL,
        avatar_cid TEXT,
        secret_key BLOB NOT NULL
    );
";

pub const CREATE_MACHINES_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS machines (
        peer_id TEXT PRIMARY KEY,
        being_id TEXT REFERENCES beings(being_id),
        machine_name TEXT NOT NULL DEFAULT 'mystery',
        secret_key BLOB NOT NULL,
        created_at INTEGER DEFAULT (unixepoch())
    );
";

pub const CREATE_AGENT_IDENTITY_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS agent_identity (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        secret_bytes BLOB NOT NULL
    );
";

pub const CREATE_MEDIA_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS media_assets (
        cid TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        local_path TEXT,          -- NULL if not downloaded yet
        download_status TEXT NOT NULL DEFAULT 'remote' -- 'remote', 'downloading', 'ready', 'failed'
    );
";

pub const CREATE_MESSAGE_ATTACHMENTS_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS message_attachments (
        message_id TEXT NOT NULL,
        cid TEXT NOT NULL,
        PRIMARY KEY (message_id, cid),
        FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
        FOREIGN KEY (cid) REFERENCES media_assets(cid) ON DELETE CASCADE
    );
";

pub const BRANCH_SELECT: &str = "
    WITH RECURSIVE branch_path AS (
        -- Base case: start with the specific message
        SELECT id, topic, sender_peer_id, parent_id, content, timestamp
        FROM messages
        WHERE id = ?

        UNION ALL

        -- Recursive step: join with the parent message
        SELECT m.id, m.topic, m.sender_peer_id, m.parent_id, m.content, m.timestamp
        FROM messages m
        JOIN branch_path bp ON m.id = bp.parent_id
    )
    SELECT * FROM branch_path ORDER BY timestamp ASC;
";

pub fn run_migrations(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
    conn.execute(CREATE_BEINGS_TABLE, [])?;
    conn.execute(CREATE_MESSAGES_TABLE, [])?;
    conn.execute(CREATE_MEDIA_TABLE, [])?;
    conn.execute(CREATE_MESSAGE_ATTACHMENTS_TABLE, [])?;
    conn.execute(CREATE_MACHINES_TABLE, [])?;
    conn.execute(CREATE_AVATAR_CACHE_TABLE, [])?;

    // Older databases have a `beings` table without the `kind` column.
    let mut stmt = conn.prepare("PRAGMA table_info(beings)")?;
    let columns: Vec<String> = stmt
        .query_map([], |row| row.get(1))?
        .collect::<rusqlite::Result<_>>()?;
    if !columns.iter().any(|c| c == "kind") {
        conn.execute(
            "ALTER TABLE beings ADD COLUMN kind TEXT NOT NULL DEFAULT 'human'",
            [],
        )?;
    }

    Ok(())
}
