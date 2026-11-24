/// Which model server an agent talks to. The local backends speak to
/// localhost servers; `Kimi` is the cloud backend (Moonshot AI,
/// OpenAI-compatible endpoint, API key from `llung_core::secrets`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// llama.cpp's llama-server on the pinned sensenova-loop fork,
    /// serving SenseNova U1.5 8B (vision-capable via --mmproj) on port
    /// 9931. Model id is auto-discovered from `/v1/models` at spawn.
    LlamaCpp,
    /// Lemonade server (OpenAI-compatible) on port 13305.
    Lemonade,
    /// Moonshot's Kimi models over HTTPS (api.moonshot.ai/v1).
    /// Platform API keys only — a "Kimi for Coding" subscription key
    /// is NOT valid here (it gets 401 Invalid Authentication).
    Kimi,
    /// "Kimi for Coding" subscription endpoint (api.kimi.com/coding/v1).
    /// Same keyring provider as `Kimi` ("kimi"), different host and
    /// model namespace (e.g. `kimi-for-coding`).
    KimiCoding,
}

impl BackendKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace(['.', '_', '-'], "").as_str() {
            "llamacpp" | "llama" => Some(Self::LlamaCpp),
            "lemonade" => Some(Self::Lemonade),
            "kimi" | "moonshot" => Some(Self::Kimi),
            "kimicoding" | "kimiforcoding" | "kfc" => Some(Self::KimiCoding),
            _ => None,
        }
    }

    /// Cloud backends need an API key before they can answer.
    pub fn needs_api_key(&self) -> bool {
        matches!(self, BackendKind::Kimi | BackendKind::KimiCoding)
    }

    /// The provider name used for key storage/lookup (`llung_core::secrets`).
    pub fn key_provider(&self) -> Option<&'static str> {
        match self {
            BackendKind::Kimi | BackendKind::KimiCoding => Some("kimi"),
            _ => None,
        }
    }
}

/// Reasoning effort for thinking models (Kimi K2 Thinking and friends).
/// Maps to a token budget in the request — see
/// `crate::interface::thinking_param`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingLevel {
    Low,
    Medium,
    High,
}

impl ThinkingLevel {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "low" | "l" => Some(Self::Low),
            "medium" | "med" | "m" => Some(Self::Medium),
            "high" | "h" => Some(Self::High),
            _ => None,
        }
    }

    /// Tokens the model may spend reasoning before answering.
    pub fn budget_tokens(&self) -> u64 {
        match self {
            ThinkingLevel::Low => 2_048,
            ThinkingLevel::Medium => 8_192,
            ThinkingLevel::High => 32_768,
        }
    }
}

impl Default for BackendKind {
    fn default() -> Self {
        Self::LlamaCpp
    }
}

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub model: String,
    pub name: String,
    pub model_url: Option<String>,
    pub backend: BackendKind,
    pub max_history: usize,
    /// Reasoning effort for thinking-model backends; None sends no
    /// thinking parameter (plain model behavior).
    pub thinking: Option<ThinkingLevel>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: crate::interface::LLAMA_FALLBACK_MODEL.into(),
            name: "llung-agent".into(),
            // No explicit override: `backend` picks the endpoint
            // (llama.cpp llama-server :9931 by default).
            model_url: None,
            backend: BackendKind::LlamaCpp,
            max_history: 20,
            thinking: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_parsing_includes_kimi() {
        assert_eq!(BackendKind::parse("kimi"), Some(BackendKind::Kimi));
        assert_eq!(BackendKind::parse("Moonshot"), Some(BackendKind::Kimi));
        assert_eq!(BackendKind::parse("llama.cpp"), Some(BackendKind::LlamaCpp));
        assert_eq!(BackendKind::parse("lemonade"), Some(BackendKind::Lemonade));
        assert_eq!(BackendKind::parse("kimi-coding"), Some(BackendKind::KimiCoding));
        assert_eq!(BackendKind::parse("KimiForCoding"), Some(BackendKind::KimiCoding));
        assert_eq!(BackendKind::parse("openai"), None);
    }

    #[test]
    fn kimi_needs_a_stored_key() {
        assert!(BackendKind::Kimi.needs_api_key());
        assert_eq!(BackendKind::Kimi.key_provider(), Some("kimi"));
        // Coding plan shares the same keyring entry as the platform key.
        assert!(BackendKind::KimiCoding.needs_api_key());
        assert_eq!(BackendKind::KimiCoding.key_provider(), Some("kimi"));
        assert!(!BackendKind::LlamaCpp.needs_api_key());
        assert_eq!(BackendKind::LlamaCpp.key_provider(), None);
    }

    #[test]
    fn thinking_levels_parse_and_have_ordered_budgets() {
        assert_eq!(ThinkingLevel::parse("low"), Some(ThinkingLevel::Low));
        assert_eq!(ThinkingLevel::parse("MED"), Some(ThinkingLevel::Medium));
        assert_eq!(ThinkingLevel::parse("high"), Some(ThinkingLevel::High));
        assert_eq!(ThinkingLevel::parse("extreme"), None);
        assert!(
            ThinkingLevel::Low.budget_tokens() < ThinkingLevel::Medium.budget_tokens()
                && ThinkingLevel::Medium.budget_tokens() < ThinkingLevel::High.budget_tokens()
        );
    }
}
