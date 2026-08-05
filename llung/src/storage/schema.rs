pub const CREATE_MESSAGES_TABLE: &str = "
    CREATE TABLE messages (
        id TEXT PRIMARY KEY,      
        topic TEXT NOT NULL,
        sender_peer_id TEXT NOT NULL,
        parent_id TEXT,           
        content TEXT NOT NULL,
        timestamp INTEGER NOT NULL
    );
";

pub const CREATE_BEINGS_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS beings (
        peer_id TEXT PRIMARY KEY,
        human_name TEXT NOT NULL,
        is_agent BOOLEAN NOT NULL,
        status TEXT NOT NULL,
        avatar_cid TEXT
    );
";

pub const CREATE_MEDIA_TABLE: &str = "
    CREATE TABLE media_assets (
        cid TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        local_path TEXT,          -- NULL if not downloaded yet
        download_status TEXT NOT NULL DEFAULT 'remote' -- 'remote', 'downloading', 'ready', 'failed'
    );
";

pub const CREATE_MESSAGE_ATTACHMENTS_TABLE: &str = "
    CREATE TABLE message_attachments (
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
        SELECT id, parent_id, content, timestamp
        FROM messages
        WHERE id = ?

        UNION ALL

        -- Recursive step: join with the parent message
        SELECT m.id, m.parent_id, m.content, m.timestamp
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
    Ok(())
}
