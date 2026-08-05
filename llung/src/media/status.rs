use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    Remote,                       // Discovered on DHT, not downloaded locally
    Queued,                       // Added to download queue
    Downloading { progress: u8 }, // Percentage 0-100
    Completed,                    // Stored locally in media folder
    Failed(String),               // Failed transfer with reason
}

impl Default for DownloadStatus {
    fn default() -> Self {
        Self::Remote
    }
}
