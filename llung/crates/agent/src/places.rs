// crates/agent/src/places.rs
//
// Where an agent works — the three filesystem roles, one source of
// truth for every tool:
//
// * workspace — the code the agent reads (and edits / writes only with
//   approval in the sandboxes). Default: the launch directory.
// * runs      — the agent's scratch area: experiment runs, checkpoints,
//   plots. Writable in every sandbox without a prompt. Default: the
//   platform data dir (`~/.local/share/llung/runs`, `~/Library/
//   Application Support/llung/runs`). Point it at an external drive for
//   big checkpoints.
// * data      — read-only input directories (CSV files, teacher traces).
//
// Resolution, per role, first match wins:
//
//   1. a runtime change from the operator (`/workspace`, `/runs`,
//      `/data` in the tui — also saved, see 3)
//   2. environment: LLUNG_WORKSPACE, LLUNG_RUNS_DIR, LLUNG_RO_BINDS
//      (dev shells / Nix wrappers; GUI apps on macOS and iOS never see
//      a shell environment, so nothing may depend on these)
//   3. the saved settings file `<config dir>/llung/places.json`
//      (machine-local on purpose: paths mean nothing on another device)
//   4. the platform default
//
// Data directories are the union of 2 and 3 (minus any the operator
// removed at runtime).
//
// Hot swapping: the resolved set lives in a process-global snapshot with
// a generation counter. Agents compare generations at the start of each
// turn (and are told what changed); REPL sessions spawned under an older
// generation are restarted so their mounts match.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

/// Where a resolved place came from (shown by `/places`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Changed this session by the operator (and saved).
    Command,
    /// From an environment variable.
    Env(&'static str),
    /// From `places.json`.
    Saved,
    /// The platform default.
    Default,
}

impl Source {
    pub fn label(&self) -> String {
        match self {
            Source::Command => "set this session, saved".into(),
            Source::Env(var) => format!("env {var}"),
            Source::Saved => "saved".into(),
            Source::Default => "default".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    pub path: PathBuf,
    pub source: Source,
}

/// The resolved places: an immutable snapshot. Get the live one with
/// [`current`]; tests and embedders build their own with [`Places::fixed`].
#[derive(Clone, Debug)]
pub struct Places {
    pub workspace: Place,
    pub runs: Place,
    pub data: Vec<Place>,
    /// Bumped on every change to the global snapshot. 0 for fixed sets.
    pub generation: u64,
}

impl Places {
    /// An explicit set (tests, embedders). Everything counts as
    /// operator-chosen, so a missing runs dir is reported, not created.
    pub fn fixed(workspace: impl Into<PathBuf>, runs: impl Into<PathBuf>, data: Vec<PathBuf>) -> Self {
        Places {
            workspace: Place { path: workspace.into(), source: Source::Command },
            runs: Place { path: runs.into(), source: Source::Command },
            data: data
                .into_iter()
                .map(|path| Place { path, source: Source::Command })
                .collect(),
            generation: 0,
        }
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace.path
    }

    pub fn runs(&self) -> &Path {
        &self.runs.path
    }

    /// The runs directory, ready to mount. The DEFAULT location is
    /// created on demand. An operator-chosen location that doesn't exist
    /// is an error, never created: it is usually an unmounted drive, and
    /// creating the mount-point directory would quietly fill the system
    /// disk with checkpoints.
    pub fn runs_ready(&self) -> Result<PathBuf, String> {
        let runs = self.runs();
        if runs.is_dir() {
            return Ok(runs.to_path_buf());
        }
        if self.runs.source == Source::Default {
            std::fs::create_dir_all(runs)
                .map_err(|e| format!("cannot create runs directory {}: {e}", runs.display()))?;
            return Ok(runs.to_path_buf());
        }
        Err(format!(
            "runs directory {} does not exist — is the drive mounted? (Not creating it: an \
             unmounted drive's mount point would silently fill the system disk.) The operator \
             can fix it with /runs <dir> or /runs reset.",
            runs.display()
        ))
    }

    /// Data directories that currently exist (missing ones — an
    /// unplugged drive — are skipped, not errors).
    pub fn data_dirs(&self) -> Vec<PathBuf> {
        self.data
            .iter()
            .filter(|p| p.path.is_dir())
            .map(|p| p.path.clone())
            .collect()
    }

    /// Everything an agent may read: workspace, runs, data.
    pub fn readable_roots(&self) -> Vec<PathBuf> {
        let mut roots = vec![self.workspace.path.clone(), self.runs.path.clone()];
        roots.extend(self.data_dirs());
        roots
    }

    /// Where host-side agent writes (the edit tool) may land: the
    /// workspace and runs. Never data.
    pub fn writable_roots(&self) -> Vec<PathBuf> {
        vec![self.workspace.path.clone(), self.runs.path.clone()]
    }

    /// Resolve an agent-named path for READING: relative paths resolve
    /// against the workspace; the target must lie under a readable root
    /// (canonicalized, so `..` and symlinks can't escape).
    pub fn readable(&self, path: &str) -> Result<PathBuf, String> {
        crate::paths::agent_readable_in(path, self.workspace(), &self.readable_roots()).map_err(
            |e| {
                if e.starts_with("outside:") {
                    format!(
                        "'{path}' is outside what this agent may read ({}). The operator can \
                         add its directory with /data add <dir>, or copy the file into the \
                         workspace.",
                        self.summary()
                    )
                } else {
                    e
                }
            },
        )
    }

    /// Resolve an agent-named path for a host-side WRITE (edit tool):
    /// workspace or runs only.
    pub fn writable(&self, path: &str) -> Result<PathBuf, String> {
        crate::paths::agent_readable_in(path, self.workspace(), &self.writable_roots()).map_err(
            |e| {
                if e.starts_with("outside:") {
                    format!(
                        "'{path}' is outside where this agent may write (workspace {}, runs {})",
                        self.workspace().display(),
                        self.runs().display()
                    )
                } else {
                    e
                }
            },
        )
    }

    /// One-line summary for error messages.
    pub fn summary(&self) -> String {
        let data = self.data_dirs();
        let data = if data.is_empty() {
            "no data dirs".to_string()
        } else {
            format!(
                "data {}",
                data.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
            )
        };
        format!(
            "workspace {}, runs {}, {data}",
            self.workspace().display(),
            self.runs().display()
        )
    }

    /// `/places` output: one line per place with its source and state.
    pub fn describe(&self) -> Vec<String> {
        let state = |p: &Path| if p.is_dir() { "" } else { "  ✗ missing" };
        let mut lines = vec![
            format!(
                "workspace  {}  ({}){}",
                self.workspace().display(),
                self.workspace.source.label(),
                state(self.workspace())
            ),
            format!(
                "runs       {}  ({}){}",
                self.runs().display(),
                self.runs.source.label(),
                match (self.runs().is_dir(), self.runs.source) {
                    (true, _) => "",
                    (false, Source::Default) => "  (created on first use)",
                    (false, _) => "  ✗ missing — drive not mounted?",
                }
            ),
        ];
        if self.data.is_empty() {
            lines.push("data       (none — /data add <dir>)".into());
        }
        for d in &self.data {
            lines.push(format!(
                "data       {}  ({}){}",
                d.path.display(),
                d.source.label(),
                state(&d.path)
            ));
        }
        lines
    }

    /// Human-readable differences from `self` to `newer` (empty when
    /// nothing an agent would notice changed).
    pub fn diff(&self, newer: &Places) -> Vec<String> {
        let mut out = Vec::new();
        if self.workspace() != newer.workspace() {
            out.push(format!(
                "workspace changed: {} → {}",
                self.workspace().display(),
                newer.workspace().display()
            ));
        }
        if self.runs() != newer.runs() {
            out.push(format!(
                "runs directory changed: {} → {}",
                self.runs().display(),
                newer.runs().display()
            ));
        }
        let old: Vec<&PathBuf> = self.data.iter().map(|p| &p.path).collect();
        let new: Vec<&PathBuf> = newer.data.iter().map(|p| &p.path).collect();
        for p in &new {
            if !old.contains(p) {
                out.push(format!("data directory added (read-only): {}", p.display()));
            }
        }
        for p in &old {
            if !new.contains(p) {
                out.push(format!("data directory removed: {}", p.display()));
            }
        }
        out
    }
}

// ---------------------------------------------------------------- layers

/// `places.json`: what the operator saved on this machine.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedPlaces {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runs: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data: Vec<PathBuf>,
}

impl SavedPlaces {
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// Atomic write (temp file + rename).
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let tmp = path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, text + "\n").map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Environment overrides, read once at startup.
#[derive(Clone, Debug, Default)]
pub struct EnvPlaces {
    pub workspace: Option<PathBuf>,
    pub runs: Option<PathBuf>,
    pub data: Vec<PathBuf>,
}

impl EnvPlaces {
    pub fn from_env() -> Self {
        let var = |k: &str| {
            std::env::var(k)
                .ok()
                .filter(|v| !v.is_empty())
                .map(|v| PathBuf::from(shellexpand::tilde(&v).as_ref()))
        };
        EnvPlaces {
            workspace: var("LLUNG_WORKSPACE"),
            runs: var("LLUNG_RUNS_DIR"),
            data: crate::paths::parse_ro_binds(std::env::var("LLUNG_RO_BINDS").ok().as_deref()),
        }
    }
}

/// Runtime changes made by the operator this session.
#[derive(Clone, Debug, Default)]
struct Overrides {
    workspace: Option<PathBuf>,
    runs: Option<PathBuf>,
    /// Data dirs removed at runtime that the environment still lists.
    hidden_data: Vec<PathBuf>,
}

/// A requested change (the tui's `/workspace`, `/runs`, `/data`).
#[derive(Clone, Debug)]
pub enum Change {
    /// `Some(dir)` sets the workspace; `None` resets to env/default.
    Workspace(Option<String>),
    /// `Some(dir)` sets the runs dir; `None` resets to env/default.
    Runs(Option<String>),
    AddData(String),
    RemoveData(String),
}

/// All inputs to resolution. Pure: [`Layers::resolve`] has no I/O, so
/// precedence is unit-testable.
#[derive(Clone, Debug)]
pub struct Layers {
    env: EnvPlaces,
    saved: SavedPlaces,
    overrides: Overrides,
    /// Launch directory (the default workspace).
    launch_dir: PathBuf,
    /// Default runs location.
    default_runs: PathBuf,
}

impl Layers {
    pub fn new(env: EnvPlaces, saved: SavedPlaces, launch_dir: PathBuf, default_runs: PathBuf) -> Self {
        Layers { env, saved, overrides: Overrides::default(), launch_dir, default_runs }
    }

    pub fn saved(&self) -> &SavedPlaces {
        &self.saved
    }

    pub fn resolve(&self, generation: u64) -> Places {
        let workspace = if let Some(p) = &self.overrides.workspace {
            Place { path: p.clone(), source: Source::Command }
        } else if let Some(p) = &self.env.workspace {
            Place { path: p.clone(), source: Source::Env("LLUNG_WORKSPACE") }
        } else if let Some(p) = &self.saved.workspace {
            Place { path: p.clone(), source: Source::Saved }
        } else {
            Place { path: self.launch_dir.clone(), source: Source::Default }
        };
        let runs = if let Some(p) = &self.overrides.runs {
            Place { path: p.clone(), source: Source::Command }
        } else if let Some(p) = &self.env.runs {
            Place { path: p.clone(), source: Source::Env("LLUNG_RUNS_DIR") }
        } else if let Some(p) = &self.saved.runs {
            Place { path: p.clone(), source: Source::Saved }
        } else {
            Place { path: self.default_runs.clone(), source: Source::Default }
        };
        let mut data: Vec<Place> = Vec::new();
        let candidates = self
            .saved
            .data
            .iter()
            .map(|p| (p, Source::Saved))
            .chain(self.env.data.iter().map(|p| (p, Source::Env("LLUNG_RO_BINDS"))));
        for (p, source) in candidates {
            if self.overrides.hidden_data.contains(p) || data.iter().any(|d| &d.path == p) {
                continue;
            }
            data.push(Place { path: p.clone(), source });
        }
        Places { workspace, runs, data, generation }
    }

    /// Apply an operator change. Returns warnings (e.g. an env var that
    /// will win again at the next launch). Validates paths against the
    /// filesystem; the caller saves `self.saved()`.
    pub fn apply(&mut self, change: Change) -> Result<Vec<String>, String> {
        let mut warnings = Vec::new();
        let current = self.resolve(0);
        match change {
            Change::Workspace(Some(raw)) => {
                let dir = existing_dir(&raw, &self.launch_dir)?;
                if dir.starts_with(current.runs()) {
                    return Err(format!(
                        "{} is inside the runs directory {} — the workspace must not live in \
                         the agent's scratch area",
                        dir.display(),
                        current.runs().display()
                    ));
                }
                if self.env.workspace.is_some() {
                    warnings.push(
                        "LLUNG_WORKSPACE is set in the environment; it will win again at the \
                         next launch"
                            .into(),
                    );
                }
                self.overrides.workspace = Some(dir.clone());
                self.saved.workspace = Some(dir);
            }
            Change::Workspace(None) => {
                self.overrides.workspace = None;
                self.saved.workspace = None;
            }
            Change::Runs(Some(raw)) => {
                let dir = existing_dir(&raw, &self.launch_dir).map_err(|e| {
                    format!("{e} — create it first (llung never creates a chosen runs directory: it may be an unmounted drive)")
                })?;
                if current.workspace().starts_with(&dir) {
                    return Err(format!(
                        "{} contains the workspace — the runs directory is writable without \
                         approval, so it must not contain your code",
                        dir.display()
                    ));
                }
                if let Some(d) = current.data_dirs().iter().find(|d| dir.starts_with(d)) {
                    warnings.push(format!(
                        "{} is inside the read-only data directory {} — it stays writable \
                         (the deeper mount wins)",
                        dir.display(),
                        d.display()
                    ));
                }
                if self.env.runs.is_some() {
                    warnings.push(
                        "LLUNG_RUNS_DIR is set in the environment; it will win again at the \
                         next launch"
                            .into(),
                    );
                }
                self.overrides.runs = Some(dir.clone());
                self.saved.runs = Some(dir);
            }
            Change::Runs(None) => {
                self.overrides.runs = None;
                self.saved.runs = None;
            }
            Change::AddData(raw) => {
                let dir = existing_dir(&raw, &self.launch_dir)?;
                if current.data.iter().any(|d| d.path == dir) {
                    return Err(format!("{} is already a data directory", dir.display()));
                }
                if current.workspace().starts_with(&dir) {
                    warnings.push(format!(
                        "{} contains the workspace — the workspace stays as it is (the deeper \
                         mount wins)",
                        dir.display()
                    ));
                }
                self.overrides.hidden_data.retain(|p| p != &dir);
                if !self.env.data.contains(&dir) {
                    self.saved.data.push(dir);
                }
            }
            Change::RemoveData(raw) => {
                let expanded = absolutize(&raw, &self.launch_dir);
                let canonical = expanded.canonicalize().unwrap_or_else(|_| expanded.clone());
                let Some(found) = current
                    .data
                    .iter()
                    .map(|d| d.path.clone())
                    .find(|p| p == &expanded || p == &canonical)
                else {
                    return Err(format!(
                        "{} is not a data directory (see /places)",
                        expanded.display()
                    ));
                };
                self.saved.data.retain(|p| p != &found);
                if self.env.data.contains(&found) {
                    self.overrides.hidden_data.push(found);
                    warnings.push(
                        "that directory comes from LLUNG_RO_BINDS; it is hidden for this \
                         session and returns at the next launch"
                            .into(),
                    );
                }
            }
        }
        Ok(warnings)
    }
}

fn absolutize(raw: &str, base: &Path) -> PathBuf {
    let p = PathBuf::from(shellexpand::tilde(raw.trim()).as_ref());
    if p.is_absolute() { p } else { base.join(p) }
}

/// `~`-expanded, absolute (relative to `base`), existing, canonical dir.
fn existing_dir(raw: &str, base: &Path) -> Result<PathBuf, String> {
    if raw.trim().is_empty() {
        return Err("a directory is required".into());
    }
    let p = absolutize(raw, base);
    if !p.is_dir() {
        return Err(format!("{} is not an existing directory", p.display()));
    }
    p.canonicalize().map_err(|e| format!("{}: {e}", p.display()))
}

// ---------------------------------------------------------------- global

/// `<config dir>/llung/places.json` — the same config directory as the
/// tui's settings.json. `None` in unit tests: tests must never read or
/// write the operator's real settings.
pub fn places_file() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    env("XDG_CONFIG_HOME")
        .or_else(|| env("HOME").map(|h| h.join(".config")))
        .map(|c| c.join("llung").join("places.json"))
}

fn default_runs_dir() -> PathBuf {
    if cfg!(test) {
        return std::env::temp_dir().join(format!("llung-test-runs-{}", std::process::id()));
    }
    llung_core::identity::session::data_dir().join("runs")
}

/// The launch directory; `/` (how GUI app bundles start) means "no
/// meaningful launch dir", so fall back to the home directory.
fn launch_dir() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
    if cwd == Path::new("/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    cwd
}

struct State {
    layers: Layers,
    current: Arc<Places>,
    /// Problem reading places.json at startup (shown by `/places`).
    load_error: Option<String>,
}

static STATE: OnceLock<RwLock<State>> = OnceLock::new();

fn state() -> &'static RwLock<State> {
    STATE.get_or_init(|| {
        let (saved, load_error) = match places_file() {
            Some(path) => match SavedPlaces::load(&path) {
                Ok(s) => (s, None),
                Err(e) => (SavedPlaces::default(), Some(e)),
            },
            None => (SavedPlaces::default(), None),
        };
        let layers = Layers::new(EnvPlaces::from_env(), saved, launch_dir(), default_runs_dir());
        let current = Arc::new(layers.resolve(1));
        RwLock::new(State { layers, current, load_error })
    })
}

/// The live places snapshot.
pub fn current() -> Arc<Places> {
    state().read().unwrap_or_else(|e| e.into_inner()).current.clone()
}

/// A problem loading places.json at startup, if any.
pub fn load_error() -> Option<String> {
    state().read().unwrap_or_else(|e| e.into_inner()).load_error.clone()
}

/// What a change did: the new snapshot, what changed (also what local
/// agents are told at their next turn), and warnings.
pub struct Report {
    pub places: Arc<Places>,
    pub changes: Vec<String>,
    pub warnings: Vec<String>,
}

/// Apply an operator change to the live places, save it, and bump the
/// generation. Agents pick it up at their next turn.
pub fn change(change: Change) -> Result<Report, String> {
    let mut st = state().write().unwrap_or_else(|e| e.into_inner());
    let mut layers = st.layers.clone();
    let mut warnings = layers.apply(change)?;
    if let Some(path) = places_file() {
        if let Err(e) = layers.saved().save(&path) {
            warnings.push(format!("applied for this session, but not saved: {e}"));
        }
    }
    let old = st.current.clone();
    let new = Arc::new(layers.resolve(old.generation + 1));
    let changes = old.diff(&new);
    st.layers = layers;
    st.current = new.clone();
    st.load_error = None;
    Ok(Report { places: new, changes, warnings })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "llung-places-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p.canonicalize().unwrap())
        }
        fn sub(&self, name: &str) -> PathBuf {
            let p = self.0.join(name);
            std::fs::create_dir_all(&p).unwrap();
            p
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    fn layers(t: &TempDir) -> Layers {
        Layers::new(EnvPlaces::default(), SavedPlaces::default(), t.sub("launch"), t.0.join("default-runs"))
    }

    #[test]
    fn precedence_is_command_env_saved_default() {
        let t = TempDir::new("prec");
        let mut l = layers(&t);
        let p = l.resolve(1);
        assert_eq!(p.workspace(), t.0.join("launch"));
        assert_eq!(p.workspace.source, Source::Default);
        assert_eq!(p.runs.source, Source::Default);

        l.saved.workspace = Some(t.sub("saved"));
        assert_eq!(l.resolve(1).workspace.source, Source::Saved);
        l.env.workspace = Some(t.sub("env"));
        assert_eq!(l.resolve(1).workspace(), t.0.join("env"));

        let warnings = l.apply(Change::Workspace(Some(t.sub("cmd").display().to_string()))).unwrap();
        assert_eq!(l.resolve(1).workspace(), t.0.join("cmd"));
        assert!(warnings[0].contains("LLUNG_WORKSPACE"), "{warnings:?}");
        assert_eq!(l.saved().workspace.as_deref(), Some(t.0.join("cmd").as_path()));

        // Reset: back to env (saved cleared too).
        l.apply(Change::Workspace(None)).unwrap();
        assert_eq!(l.resolve(1).workspace(), t.0.join("env"));
        assert!(l.saved().workspace.is_none());
    }

    #[test]
    fn chosen_runs_dir_must_exist_and_is_never_created() {
        let t = TempDir::new("runs");
        let mut l = layers(&t);
        let missing = t.0.join("unmounted-drive/runs");
        let err = l.apply(Change::Runs(Some(missing.display().to_string()))).unwrap_err();
        assert!(err.contains("create it first"), "{err}");
        assert!(!missing.exists());

        // A saved runs dir whose drive is gone: runs_ready refuses.
        l.saved.runs = Some(missing.clone());
        let err = l.resolve(1).runs_ready().unwrap_err();
        assert!(err.contains("drive mounted"), "{err}");
        assert!(!missing.exists(), "never create a chosen runs dir");

        // The default IS created on demand.
        l.saved.runs = None;
        let ready = l.resolve(1).runs_ready().unwrap();
        assert!(ready.is_dir());
    }

    #[test]
    fn runs_must_not_contain_the_workspace() {
        let t = TempDir::new("contain");
        let mut l = layers(&t);
        // launch dir (workspace) is t/launch; t itself contains it.
        let err = l.apply(Change::Runs(Some(t.0.display().to_string()))).unwrap_err();
        assert!(err.contains("contains the workspace"), "{err}");
        // A runs dir INSIDE the workspace is fine (experiments/runs style).
        let inner = t.sub("launch/experiments/runs");
        l.apply(Change::Runs(Some(inner.display().to_string()))).unwrap();
        assert_eq!(l.resolve(1).runs(), inner);
        // ...but then the workspace can't move into the runs dir.
        let err = l
            .apply(Change::Workspace(Some(t.sub("launch/experiments/runs/x").display().to_string())))
            .unwrap_err();
        assert!(err.contains("inside the runs directory"), "{err}");
    }

    #[test]
    fn data_dirs_union_add_remove_and_env_hiding() {
        let t = TempDir::new("data");
        let mut l = layers(&t);
        let env_dir = t.sub("env-data");
        l.env.data = vec![env_dir.clone()];
        let added = t.sub("added");
        l.apply(Change::AddData(added.display().to_string())).unwrap();
        let p = l.resolve(1);
        assert_eq!(p.data.len(), 2);
        assert_eq!(l.saved().data, vec![added.clone()], "env dirs are not copied into the file");
        assert!(l.apply(Change::AddData(added.display().to_string())).is_err(), "no duplicates");

        l.apply(Change::RemoveData(added.display().to_string())).unwrap();
        assert!(l.saved().data.is_empty());
        let warnings = l.apply(Change::RemoveData(env_dir.display().to_string())).unwrap();
        assert!(warnings[0].contains("LLUNG_RO_BINDS"));
        assert!(l.resolve(1).data.is_empty(), "hidden for the session");
        assert!(l.apply(Change::RemoveData("/nope".into())).is_err());

        // Re-adding an env dir un-hides it without saving a duplicate.
        l.apply(Change::AddData(env_dir.display().to_string())).unwrap();
        assert_eq!(l.resolve(1).data_dirs(), vec![env_dir]);
        assert!(l.saved().data.is_empty());
    }

    #[test]
    fn saved_file_round_trips() {
        let t = TempDir::new("file");
        let path = t.0.join("cfg/places.json");
        assert_eq!(SavedPlaces::load(&path).unwrap(), SavedPlaces::default());
        let s = SavedPlaces {
            workspace: Some("/w".into()),
            runs: None,
            data: vec!["/d".into()],
        };
        s.save(&path).unwrap();
        assert_eq!(SavedPlaces::load(&path).unwrap(), s);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("runs"), "unset keys are omitted: {text}");
        std::fs::write(&path, "not json").unwrap();
        assert!(SavedPlaces::load(&path).is_err());
    }

    #[test]
    fn readable_and_writable_rules() {
        let t = TempDir::new("rules");
        let ws = t.sub("ws");
        let runs = t.sub("runs");
        let data = t.sub("data");
        let secret = t.sub("secret");
        for (dir, f) in [(&ws, "a.txt"), (&runs, "r.txt"), (&data, "d.csv"), (&secret, "s")] {
            std::fs::write(dir.join(f), "x").unwrap();
        }
        let p = Places::fixed(&ws, &runs, vec![data.clone()]);
        assert!(p.readable("a.txt").is_ok(), "relative = workspace");
        assert!(p.readable(runs.join("r.txt").to_str().unwrap()).is_ok());
        assert!(p.readable(data.join("d.csv").to_str().unwrap()).is_ok());
        let err = p.readable(secret.join("s").to_str().unwrap()).unwrap_err();
        assert!(err.contains("outside what this agent may read"), "{err}");
        assert!(err.contains("/data add"), "{err}");

        assert!(p.writable("a.txt").is_ok());
        assert!(p.writable(runs.join("r.txt").to_str().unwrap()).is_ok());
        let err = p.writable(data.join("d.csv").to_str().unwrap()).unwrap_err();
        assert!(err.contains("outside where this agent may write"), "{err}");
    }

    #[test]
    fn diff_lists_what_changed() {
        let a = Places::fixed("/w1", "/r", vec!["/d1".into()]);
        let b = Places::fixed("/w2", "/r", vec!["/d2".into()]);
        let d = a.diff(&b);
        assert_eq!(d.len(), 3, "{d:?}");
        assert!(d[0].contains("/w1 → /w2"));
        assert!(d.iter().any(|l| l.contains("added") && l.contains("/d2")));
        assert!(d.iter().any(|l| l.contains("removed") && l.contains("/d1")));
        assert!(a.diff(&a.clone()).is_empty());
    }
}
