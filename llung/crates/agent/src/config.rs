/// Which model server an agent talks to. The local backends speak to
/// localhost servers; `Kimi`/`KimiCoding` (Moonshot AI, OpenAI-compatible)
/// and `Anthropic` (native Messages API) are cloud backends with an API
/// key from `llung_core::secrets`.
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
    /// Anthropic's Claude models over the native Messages API
    /// (api.anthropic.com/v1/messages) — `interface::Anthropic`, not
    /// `OpenAiCompatible`. Keyring provider "anthropic".
    Anthropic,
}

impl BackendKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace(['.', '_', '-'], "").as_str() {
            "llamacpp" | "llama" => Some(Self::LlamaCpp),
            "lemonade" => Some(Self::Lemonade),
            "kimi" | "moonshot" => Some(Self::Kimi),
            "kimicoding" | "kimiforcoding" | "kfc" => Some(Self::KimiCoding),
            "anthropic" | "claude" => Some(Self::Anthropic),
            _ => None,
        }
    }

    /// Cloud backends need an API key before they can answer.
    pub fn needs_api_key(&self) -> bool {
        self.key_provider().is_some()
    }

    /// The provider name used for key storage/lookup (`llung_core::secrets`).
    pub fn key_provider(&self) -> Option<&'static str> {
        match self {
            BackendKind::Kimi | BackendKind::KimiCoding => Some("kimi"),
            BackendKind::Anthropic => Some("anthropic"),
            BackendKind::LlamaCpp | BackendKind::Lemonade => None,
        }
    }

    /// Every keyring provider (`/key list`): model backends, plus
    /// `github` (optional token for web_fetch: rate limits, private repos).
    pub const KEY_PROVIDERS: &'static [&'static str] = &["kimi", "anthropic", "github"];

    /// Model id when a spawn line names none.
    pub fn default_model(&self) -> String {
        match self {
            BackendKind::Kimi => "kimi-k2-0905-preview".into(),
            BackendKind::KimiCoding => "kimi-for-coding".into(),
            BackendKind::Anthropic => crate::interface::ANTHROPIC_DEFAULT_MODEL.into(),
            BackendKind::LlamaCpp | BackendKind::Lemonade => AgentConfig::default().model,
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

/// How the harness drives a class of model: context size, tool-loop
/// limits, prompt shape. Derived from the backend, not configured per
/// spawn — small local models and frontier cloud models need opposite
/// trade-offs (a small model drowns in a long transcript; a frontier
/// model loses the plot without one).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelProfile {
    /// Byte budget of the history view (`HistoryCompressor::context_budget`).
    pub context_budget: usize,
    /// Tool executions per turn before the turn is handed back.
    pub max_tool_steps: u32,
    /// Cap (chars) on ONE tool result fed back into the prompt.
    pub tool_feedback_chars: usize,
    /// Feed the turn's whole tool transcript back (assistant call /
    /// user result pairs). Off: only the latest result plus one-line
    /// summaries of earlier steps (what small models handle best).
    pub full_transcript: bool,
    /// Cap (chars) on the whole in-turn transcript; older results are
    /// elided first. Only with `full_transcript`.
    pub transcript_chars: usize,
    /// Attach images the agent published this turn (`show_image`) to
    /// the tool result, so a vision model sees its own plots. Only with
    /// `full_transcript`.
    pub see_own_images: bool,
}

impl ModelProfile {
    /// Small local models (SenseNova, LFM) — and the default.
    pub fn small_model() -> Self {
        Self {
            context_budget: crate::history::DEFAULT_CONTEXT_BUDGET,
            max_tool_steps: 8,
            tool_feedback_chars: 4_000,
            full_transcript: false,
            transcript_chars: 0,
            see_own_images: false,
        }
    }

    /// Frontier models with large context windows (Claude, 200k tokens).
    /// ~400 KB of history (~100k tokens) stays verbatim — no summarizer
    /// calls until a topic outgrows it.
    pub fn frontier() -> Self {
        Self {
            context_budget: 400_000,
            max_tool_steps: 24,
            tool_feedback_chars: 16_000,
            full_transcript: true,
            transcript_chars: 120_000,
            see_own_images: true,
        }
    }

    /// The profile for a backend, with `LLUNG_CONTEXT_BUDGET` (bytes)
    /// and `LLUNG_MAX_TOOL_STEPS` overrides applied — e.g. a smaller
    /// budget to cap what a long topic costs per turn on a paid model.
    pub fn for_backend(kind: BackendKind) -> Self {
        let mut profile = match kind {
            BackendKind::Anthropic => Self::frontier(),
            // Kimi keeps the small-model shape it was tuned with.
            BackendKind::LlamaCpp
            | BackendKind::Lemonade
            | BackendKind::Kimi
            | BackendKind::KimiCoding => Self::small_model(),
        };
        let env = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok());
        if let Some(bytes) = env("LLUNG_CONTEXT_BUDGET").filter(|b| *b > 0) {
            profile.context_budget = bytes as usize;
        }
        if let Some(steps) = env("LLUNG_MAX_TOOL_STEPS").filter(|s| *s > 0) {
            profile.max_tool_steps = steps.min(u32::MAX as u64) as u32;
        }
        profile
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
        assert_eq!(BackendKind::parse("anthropic"), Some(BackendKind::Anthropic));
        assert_eq!(BackendKind::parse("Claude"), Some(BackendKind::Anthropic));
        assert_eq!(BackendKind::parse("openai"), None);
    }

    #[test]
    fn kimi_needs_a_stored_key() {
        assert!(BackendKind::Kimi.needs_api_key());
        assert_eq!(BackendKind::Kimi.key_provider(), Some("kimi"));
        // Coding plan shares the same keyring entry as the platform key.
        assert!(BackendKind::KimiCoding.needs_api_key());
        assert_eq!(BackendKind::KimiCoding.key_provider(), Some("kimi"));
        assert!(BackendKind::Anthropic.needs_api_key());
        assert_eq!(BackendKind::Anthropic.key_provider(), Some("anthropic"));
        for kind in [BackendKind::Kimi, BackendKind::Anthropic] {
            let provider = kind.key_provider().unwrap();
            assert!(BackendKind::KEY_PROVIDERS.contains(&provider), "{provider} unlisted");
        }
        assert!(!BackendKind::LlamaCpp.needs_api_key());
        assert_eq!(BackendKind::LlamaCpp.key_provider(), None);
    }

    #[test]
    fn profiles_follow_the_backend() {
        if std::env::var("LLUNG_CONTEXT_BUDGET").is_ok() || std::env::var("LLUNG_MAX_TOOL_STEPS").is_ok() {
            return; // overrides set in this environment
        }
        assert_eq!(ModelProfile::for_backend(BackendKind::Anthropic), ModelProfile::frontier());
        assert_eq!(ModelProfile::for_backend(BackendKind::LlamaCpp), ModelProfile::small_model());
        let f = ModelProfile::frontier();
        assert!(f.context_budget > ModelProfile::small_model().context_budget);
        assert!(f.transcript_chars >= f.tool_feedback_chars, "one result must fit the transcript");
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
