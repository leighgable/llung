//! Input-completion picker: pi-style selectable lists derived from the
//! input buffer — model/backend pickers for `/agent spawn` options and a
//! fuzzy file finder for `@…` mentions and `/attach` paths.
//!
//! Everything here is a pure function of `(text, cursor)`: the app
//! re-derives the picker on every keystroke, so there is no state to
//! keep in sync with the buffer. `None` means "no picker" — the Taffy
//! popup node collapses to zero height and the UI is exactly as before.

use agent::config::BackendKind;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use crate::theme::{Role, Theme};

/// Max item rows the picker shows; the list scrolls a window around
/// the selection beyond this.
pub const MAX_VISIBLE: usize = 8;
/// Cap scanned directory entries (score+sort happens on at most this
/// many candidates; big directories stay cheap).
const SCAN_CAP: usize = 100;

/// One selectable entry in the picker.
#[derive(Clone, Debug)]
pub struct Item {
    /// Main text (model id, backend name, file name, …).
    pub label: String,
    /// Right-aligned annotation (endpoint, token budget, byte size, …).
    pub detail: String,
    /// Text substituted for the completed token on accept.
    pub replace: String,
    /// Directories and bare keys (`backend=`) keep the picker open to
    /// complete the next level; finished values close it.
    pub keep_open: bool,
}

/// The picker state for one completable token in the input buffer.
#[derive(Clone, Debug)]
pub struct Completion {
    /// Byte range of the input token that accepting replaces.
    pub replace_start: usize,
    pub replace_end: usize,
    /// Box title: the completion context (e.g. `spawn · model (kimi)`,
    /// or the directory being listed).
    pub title: String,
    pub items: Vec<Item>,
    pub selected: usize,
}

impl Completion {
    /// Taffy rows the popup node needs: a title border row + items.
    pub fn visible_rows(&self) -> usize {
        1 + self.items.len().min(MAX_VISIBLE)
    }

    /// Wrap-around selection movement (Up/Down while the picker is open).
    pub fn move_selection(&mut self, delta: isize) {
        let len = self.items.len();
        if len == 0 {
            return;
        }
        self.selected = (self.selected as isize + delta).rem_euclid(len as isize) as usize;
    }
}

/// Derive the picker for the buffer, or `None` when nothing is
/// completable. `dismissed_for` is the buffer text as of the last
/// Esc-dismissal: while the text is unchanged the picker stays hidden
/// (so a second Esc reaches the app's quit handler).
pub fn for_input(text: &str, cursor: usize, dismissed_for: &str) -> Option<Completion> {
    if text == dismissed_for || text.trim().is_empty() {
        return None;
    }
    let (start, end) = token_span(text, cursor);
    let token = &text[start..end];
    file_completion(text, token, start, end)
        .or_else(|| spawn_completion(text, token, start, end))
}

/// Byte span of the whitespace-delimited token under the cursor.
/// `cursor` is always on a char boundary (`crate::input::InputLine`
/// maintains that invariant).
fn token_span(text: &str, cursor: usize) -> (usize, usize) {
    let start = text[..cursor].rfind(' ').map(|i| i + 1).unwrap_or(0);
    let end = text[cursor..]
        .find(' ')
        .map(|i| cursor + i)
        .unwrap_or(text.len());
    (start, end)
}

// ---------------------------------------------------------------------------
// File completion: `@…` mentions and `/attach` / `/plot` paths.
// ---------------------------------------------------------------------------

fn file_completion(text: &str, token: &str, start: usize, end: usize) -> Option<Completion> {
    let mention = token.starts_with('@');
    let command = text.split(' ').next().unwrap_or("");
    let token_idx = text[..start].split(' ').filter(|t| !t.is_empty()).count();
    let command_path = matches!(command, "/attach" | "/plot") && token_idx == 1;
    if !mention && !command_path {
        return None;
    }

    let quoted = token.starts_with('"') || (command_path && token.starts_with('"'));
    let raw = token
        .strip_prefix('@')
        .unwrap_or(token)
        .trim_start_matches('"');
    // Require an absolute-ish anchor (`@`, `@/`, `@~`, `@"…`) so a bare
    // word like `@plotter` (an agent mention) never opens the finder.
    if !(raw.is_empty() || raw.starts_with('/') || raw.starts_with('~')) {
        return None;
    }

    let (lookup_dir, display_dir, frag) = split_path(raw)?;
    let prefix = if mention { "@" } else if quoted { "\"" } else { "" };
    let title = if mention {
        format!("mention · {display_dir}")
    } else {
        format!("{command} · {display_dir}")
    };
    list_dir(&lookup_dir, &display_dir, frag, prefix, title, start, end)
}

/// File picker for a form field that holds nothing but a path (the
/// registration avatar): the entire string is completed at once, so
/// `replace` covers `0..len`. Relative fragments list the launch
/// directory — the same resolution the avatar field applies.
pub fn for_path_field(text: &str) -> Option<Completion> {
    let (lookup_dir, display_dir, frag, title) = if text.is_empty() {
        let cwd = std::env::current_dir().ok()?;
        (
            cwd.to_string_lossy().into_owned() + "/",
            // Replacement stays relative ("name/"), only the title
            // orients the user with the absolute directory.
            String::new(),
            "",
            format!("file · {}/", cwd.display()),
        )
    } else if text.starts_with('/') || text.starts_with('~') {
        let (lookup, display, frag) = split_path(text)?;
        (lookup, display, frag, String::new())
    } else {
        // Relative path: complete against the launch directory so the
        // picked entry lands exactly where the field would look it up.
        let cwd = std::env::current_dir().ok()?;
        let cut = text.rfind('/').map(|i| i + 1).unwrap_or(0);
        let (parent, frag) = text.split_at(cut);
        (
            cwd.join(parent).to_string_lossy().into_owned(),
            parent.to_string(),
            frag,
            String::new(),
        )
    };
    let title = if title.is_empty() {
        format!("file · {display_dir}")
    } else {
        title
    };
    list_dir(&lookup_dir, &display_dir, frag, "", title, 0, text.len())
}

/// List one directory and build the picker for it. Shared by the
/// chat-input token completion and the bare-path form field.
#[allow(clippy::too_many_arguments)]
fn list_dir(
    lookup_dir: &str,
    display_dir: &str,
    frag: &str,
    prefix: &str,
    title: String,
    start: usize,
    end: usize,
) -> Option<Completion> {
    let lookup = std::path::Path::new(lookup_dir);

    // A fragment that already resolves to an existing file is complete —
    // no picker (Enter should submit/send, not re-pick).
    if !frag.is_empty() && lookup.join(frag).is_file() {
        return None;
    }

    let mut scored: Vec<(u32, bool, u64, String)> = Vec::new();
    for entry in std::fs::read_dir(lookup).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        // Dotfiles only surface when the fragment asks for them.
        if name.starts_with('.') && !frag.starts_with('.') {
            continue;
        }
        let Some(score) = fuzzy(&name, frag) else {
            continue;
        };
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let size = if is_dir { 0 } else { entry.metadata().map(|m| m.len()).unwrap_or(0) };
        scored.push((score, is_dir, size, name));
        if scored.len() >= SCAN_CAP {
            break;
        }
    }
    if scored.is_empty() {
        return None;
    }
    // Directories first (navigation beats files), then score, then name.
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)).then(a.3.cmp(&b.3)));

    let items = scored
        .into_iter()
        .map(|(_, is_dir, size, name)| {
            let slash = if is_dir { "/" } else { "" };
            Item {
                label: format!("{name}{slash}"),
                detail: if is_dir { "dir".into() } else { human_size(size) },
                replace: format!("{prefix}{display_dir}{name}{slash}"),
                keep_open: is_dir,
            }
        })
        .collect();

    Some(Completion {
        replace_start: start,
        replace_end: end,
        title,
        items,
        selected: 0,
    })
}

/// Split a partially typed path into `(lookup_dir, display_dir, fragment)`.
/// Both dirs carry a trailing slash; `lookup` is tilde-expanded (for
/// `std::fs`), `display` preserves the user's `~` (so the replacement
/// keeps working with `expand_file_mentions`). `None` when the fragment
/// isn't anchored to a directory we can list.
fn split_path(raw: &str) -> Option<(String, String, &str)> {
    if raw.is_empty() || raw == "~" || raw.starts_with("~/") {
        let home = std::env::var("HOME").ok().filter(|h| !h.is_empty())?;
        let after = raw.strip_prefix('~').unwrap_or("");
        let cut = after.rfind('/').map(|i| i + 1).unwrap_or(0);
        let (dir, frag) = after.split_at(cut);
        let dir = if dir.is_empty() { "/" } else { dir };
        return Some((format!("{home}{dir}"), format!("~{dir}"), frag));
    }
    let cut = raw.rfind('/')? + 1;
    let (dir, frag) = raw.split_at(cut);
    Some((dir.to_string(), dir.to_string(), frag))
}

/// 950 → "950B", 3456 → "3.5kB", 1_234_567 → "1.2MB".
fn human_size(n: u64) -> String {
    if n < 1_000 {
        format!("{n}B")
    } else if n < 1_000_000 {
        format!("{:.1}kB", n as f64 / 1_000.0)
    } else {
        format!("{:.1}MB", n as f64 / 1_000_000.0)
    }
}

// ---------------------------------------------------------------------------
// /agent spawn completion: backend= / model= / thinking= values.
// ---------------------------------------------------------------------------

const BACKENDS: &[(&str, &str)] = &[
    ("llamacpp", "local llama.cpp :9931 (SenseNova 9B)"),
    ("lemonade", "local Lemonade :13305 (LFM2.5)"),
    ("kimi", "Moonshot cloud · needs API key"),
    ("kimi-coding", "Kimi for Coding · needs API key"),
];

/// Static model catalog per backend. Local servers host exactly one
/// model each; cloud catalogs list the ids worth typing.
fn models_for(kind: BackendKind) -> &'static [(&'static str, &'static str)] {
    match kind {
        BackendKind::LlamaCpp => &[("sensenova-9b", "local llama.cpp :9931 (SenseNova U1.5 8B)")],
        BackendKind::Lemonade => &[("LFM2.5-8B-A1B", "local Lemonade :13305")],
        BackendKind::Kimi => &[
            ("kimi-k2-0905-preview", "K2, 256k ctx"),
            ("kimi-k2-turbo-preview", "K2 fast variant"),
            ("kimi-k2-thinking", "reasoning model"),
            ("moonshot-v1-128k", "older generation, 128k"),
        ],
        BackendKind::KimiCoding => &[("kimi-for-coding", "coding subscription")],
    }
}

const THINKING: &[(&str, &str)] = &[
    ("low", "2k reasoning budget"),
    ("medium", "8k reasoning budget"),
    ("high", "32k reasoning budget"),
];

fn spawn_completion(text: &str, token: &str, start: usize, end: usize) -> Option<Completion> {
    let toks: Vec<&str> = text.split(' ').filter(|t| !t.is_empty()).collect();
    if toks.first().copied() != Some("/agent") || toks.get(1).copied() != Some("spawn") {
        return None;
    }
    // Token 2 is the agent's name — never complete that.
    let idx = text[..start].split(' ').filter(|t| !t.is_empty()).count();
    if idx < 3 {
        return None;
    }

    // `prefix` is the `key=` part of the token being replaced: value
    // acceptance must reproduce it (`backend=lem` -> `backend=lemonade `),
    // never bare `lemonade ` — the latter is a syntax error downstream.
    let build = |title: String,
                 cands: &[(&str, &str)],
                 frag: &str,
                 prefix: &str,
                 suffix: &str,
                 keep_open: bool| {
        let items: Vec<Item> = cands
            .iter()
            .filter(|(name, _)| fuzzy(name, frag).is_some())
            .map(|(name, detail)| Item {
                label: name.to_string(),
                detail: detail.to_string(),
                replace: format!("{prefix}{name}{suffix}"),
                keep_open,
            })
            .collect();
        (!items.is_empty()).then_some(Completion {
            replace_start: start,
            replace_end: end,
            title,
            items,
            selected: 0,
        })
    };

    match token.split_once('=') {
        // Value position: `backend=lem` → backend list, and so on.
        Some(("backend", frag)) => build(
            "spawn · backend".into(),
            BACKENDS,
            frag,
            "backend=",
            " ",
            false,
        ),
        Some(("model", frag)) => {
            // Models depend on the backend already typed (default:
            // llamacpp) — `/agent spawn scout backend=kimi model=` lists
            // Kimi ids, not SenseNova.
            let kind = toks
                .iter()
                .find_map(|t| t.strip_prefix("backend="))
                .and_then(BackendKind::parse)
                .unwrap_or_default();
            build(
                format!("spawn · model ({kind:?})"),
                models_for(kind),
                frag,
                "model=",
                " ",
                false,
            )
        }
        Some(("thinking", frag)) => build(
            "spawn · thinking".into(),
            THINKING,
            frag,
            "thinking=",
            " ",
            false,
        ),
        // Key position: a bare fragment (or nothing) after the name
        // suggests the option keys; accepting one opens its value list.
        None => {
            let keys: &[(&str, &str)] = &[
                ("backend=", "model server"),
                ("model=", "model id (per backend)"),
                ("thinking=", "reasoning effort"),
            ];
            build("spawn · options".into(), keys, token, "", "", true)
        }
        Some(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Fuzzy filter: substring first (earlier + shorter wins), subsequence
// fallback. No new dependency — the catalogs and directories are small.
// ---------------------------------------------------------------------------

fn fuzzy(candidate: &str, frag: &str) -> Option<u32> {
    if frag.is_empty() {
        return Some(0);
    }
    let f = frag.to_ascii_lowercase();
    let c = candidate.to_ascii_lowercase();
    if let Some(i) = c.find(&f) {
        // Prefix matches score above later substrings; shorter names win.
        return Some(u32::MAX - 1000 - (i as u32) * 10 - candidate.len() as u32);
    }
    // Greedy subsequence match, penalized by skipped characters.
    let mut it = c.chars();
    let mut cost = 0u32;
    for fc in f.chars() {
        loop {
            let cc = it.next()?;
            if cc == fc {
                break;
            }
            cost += 1;
        }
    }
    Some(u32::MAX / 2 - cost)
}

// ---------------------------------------------------------------------------
// Rendering: shared by the chat popup strip (Taffy node above the
// input) and the registration form's avatar field.
// ---------------------------------------------------------------------------

/// Render the picker into its own rect: a title border row
/// (" spawn · backend ") then the selectable items with the detail
/// column right-aligned. Draws only `area` — whatever surrounds it
/// renders exactly as it would without a picker.
pub fn render_picker(buf: &mut Buffer, area: Rect, comp: &Completion, theme: &Theme) {
    if area.height == 0 || area.width < 6 {
        return;
    }
    use unicode_width::UnicodeWidthStr;
    let panel_bg = theme.bg(Role::Panel);
    let buf_area = *buf.area();

    let x0 = area.x.min(buf_area.width);
    let y0 = area.y.min(buf_area.height);
    let x1 = (area.x + area.width).min(buf_area.width);
    let y1 = (area.y + area.height).min(buf_area.height);

    for y in y0..y1 {
        for x in x0..x1 {
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.reset();
                cell.set_bg(panel_bg);
            }
        }
    }

    // Title row: a border line with the completion context embedded.
    let border = theme.style(Role::BorderAccent);
    for x in x0..x1 {
        if let Some(cell) = buf.cell_mut((x, y0)) {
            cell.set_symbol("─");
            cell.set_style(border);
        }
    }
    let title = format!(" {} ", comp.title);
    buf.set_stringn(
        x0 + 1,
        y0,
        &title,
        (x1 - x0).saturating_sub(2) as usize,
        theme.style(Role::PanelTitle),
    );

    // Item rows, scrolled so the selection stays visible.
    let rows = (y1 - y0).saturating_sub(1) as usize;
    if rows == 0 {
        return;
    }
    let start = if comp.selected >= rows {
        comp.selected + 1 - rows
    } else {
        0
    };
    for (row, item) in comp.items[start..].iter().take(rows).enumerate() {
        let y = y0 + 1 + row as u16;
        let selected = start + row == comp.selected;
        let style = if selected {
            theme.style(Role::ActiveItem)
        } else {
            theme.style(Role::InactiveItem)
        };
        buf.set_stringn(x0 + 1, y, "› ", (x1 - x0 - 1) as usize, style);
        // Reserve the rightmost 16 columns for the detail.
        let label_max = (x1.saturating_sub(x0 + 3 + 16)) as usize;
        buf.set_stringn(x0 + 3, y, &item.label, label_max, style);
        if !item.detail.is_empty() {
            let dw = item.detail.width() as u16;
            let dx = x1.saturating_sub(1 + dw);
            if dx > x0 + 3 {
                buf.set_stringn(
                    dx,
                    y,
                    &item.detail,
                    (x1 - 1 - dx) as usize,
                    theme.style(Role::Dim),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(c: &Completion) -> Vec<&str> {
        c.items.iter().map(|i| i.label.as_str()).collect()
    }

    #[test]
    fn spawn_suggests_option_keys_after_the_name() {
        let text = "/agent spawn scout ";
        let c = for_input(text, text.len(), "").unwrap();
        assert_eq!(c.title, "spawn · options");
        assert_eq!(labels(&c), vec!["backend=", "model=", "thinking="]);
        // Accepting a key keeps the picker open for its values.
        assert!(c.items.iter().all(|i| i.keep_open));
    }

    #[test]
    fn spawn_completes_backend_values() {
        let text = "/agent spawn scout backend=lem";
        let c = for_input(text, text.len(), "").unwrap();
        assert_eq!(c.title, "spawn · backend");
        assert_eq!(labels(&c), vec!["lemonade"]);
    }

    #[test]
    fn backend_value_acceptance_replaces_the_whole_token() {
        let text = "/agent spawn scout backend=lem";
        let c = for_input(text, text.len(), "").unwrap();
        let (s, e) = (c.replace_start, c.replace_end);
        assert_eq!(&text[s..e], "backend=lem");
        // The key prefix must survive acceptance: bare `lemonade ` is a
        // syntax error downstream (same for a fully-typed value).
        assert_eq!(c.items[0].replace, "backend=lemonade ");
        assert!(!c.items[0].keep_open);

        // Even with the value fully typed, accepting rewrites the token
        // WITH its prefix (previously it stripped `backend=`).
        let text = "/agent spawn sensei backend=llamacpp";
        let c = for_input(text, text.len(), "").unwrap();
        assert_eq!(c.items[0].replace, "backend=llamacpp ");
    }

    #[test]
    fn model_list_follows_the_typed_backend() {
        let text = "/agent spawn scout backend=kimi model=";
        let c = for_input(text, text.len(), "").unwrap();
        assert!(c.title.contains("Kimi"), "{c:?}");
        assert!(labels(&c).contains(&"kimi-k2-0905-preview"));

        // No backend typed yet: the default backend's single model.
        let text = "/agent spawn scout model=";
        let c = for_input(text, text.len(), "").unwrap();
        assert_eq!(labels(&c), vec!["sensenova-9b"]);
        assert_eq!(c.items[0].replace, "model=sensenova-9b ");
    }

    #[test]
    fn thinking_filters_by_fragment() {
        let text = "/agent spawn scout thinking=h";
        let c = for_input(text, text.len(), "").unwrap();
        assert_eq!(labels(&c), vec!["high"]);
        assert!(c.items[0].replace.starts_with("thinking=high "));
    }

    #[test]
    fn the_name_token_is_not_completed() {
        assert!(for_input("/agent spawn he", 15, "").is_none());
    }

    #[test]
    fn outside_agent_spawn_nothing_completes() {
        assert!(for_input("/join topic", 11, "").is_none());
        assert!(for_input("hello world", 11, "").is_none());
    }

    #[test]
    fn dismissed_buffer_stays_hidden_until_it_changes() {
        let text = "/agent spawn scout backend=";
        assert!(for_input(text, text.len(), "").is_some());
        // Same text after an Esc: hidden. One char more: recompute.
        assert!(for_input(text, text.len(), text).is_none());
        let longer = format!("{text}k");
        assert!(for_input(&longer, longer.len(), text).is_some());
    }

    #[test]
    fn mention_opens_finder_and_directories_stay_open() {
        let dir = std::env::temp_dir().join("llung-completion-test");
        std::fs::create_dir_all(dir.join("subdir")).unwrap();
        std::fs::write(dir.join("data.csv"), "a,b\n").unwrap();

        // Trailing slash: the finder lists the directory's entries.
        let text = format!("plot @{}/", dir.display());
        let c = for_input(&text, text.len(), "").unwrap();
        assert!(c.title.starts_with("mention ·"), "{c:?}");
        // Directory sorts first and keeps the picker open to descend.
        assert_eq!(c.items[0].label, "subdir/");
        assert!(c.items[0].keep_open);
        assert!(c.items[0].replace.ends_with("subdir/"));
        // Files carry a size annotation and close the picker.
        let csv = c.items.iter().find(|i| i.label == "data.csv").unwrap();
        assert_eq!(csv.detail, "4B");
        assert!(!csv.keep_open);

        // Completing the exact file name yields no picker.
        let text = format!("plot @{}/data.csv", dir.display());
        assert!(for_input(&text, text.len(), "").is_none());

        // Without the slash the finder lists the PARENT (here /tmp),
        // filtered by the fragment — accepting the dir descends into it.
        let text = format!("plot @{}", dir.display());
        let c = for_input(&text, text.len(), "").unwrap();
        let dir_item = c.items.iter().find(|i| i.label == "llung-completion-test/").unwrap();
        assert!(dir_item.keep_open);
        assert!(dir_item.replace.ends_with("llung-completion-test/"));

        // Fragment filters the directory listing.
        let text = format!("plot @{}/su", dir.display());
        let c = for_input(&text, text.len(), "").unwrap();
        assert_eq!(labels(&c), vec!["subdir/"]);

        // Agent mentions (no path anchor) never open the finder.
        assert!(for_input("hey @plotter", 12, "").is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn attach_command_completes_its_path_argument() {
        let dir = std::env::temp_dir().join("llung-attach-test");
        std::fs::create_dir_all(&dir).unwrap();
        let text = format!("/attach {}", dir.display());
        let c = for_input(&text, text.len(), "").unwrap();
        assert!(c.title.starts_with("/attach ·"), "{c:?}");
        assert!(!c.items[0].replace.starts_with('@'));

        // The caption argument (second token) is not completed.
        let text = format!("/attach {}/x pic", dir.display());
        let c = for_input(&text, text.len(), "");
        assert!(c.is_none(), "{c:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn path_field_lists_cwd_and_descends_into_dirs() {
        // The whole field is the path: an empty field lists the launch
        // directory (cwd during tests = the crate dir), replacement
        // stays relative.
        let c = for_path_field("").expect("empty field should list cwd");
        assert!(c.title.contains("file · /"), "{c:?}");
        assert!(c.items.iter().any(|i| i.label == "src/"));
        assert!(c.items.iter().any(|i| i.label == "Cargo.toml"));
        assert_eq!((c.replace_start, c.replace_end), (0, 0));
        let src = c.items.iter().find(|i| i.label == "src/").unwrap();
        assert!(src.keep_open);
        assert_eq!(src.replace, "src/");

        // A directory fragment lists the parent; picking descends.
        let c = for_path_field("sr").unwrap();
        assert_eq!(c.items[0].label, "src/");
        let c = for_path_field("src/").unwrap();
        assert!(c.title.contains("file · src/"), "{c:?}");
        assert!(c.items.iter().all(|i| i.keep_open == i.label.ends_with('/')));

        // A fragment that resolves to an existing file is complete.
        assert!(for_path_field("Cargo.toml").is_none());

        // Anchored paths list their directory.
        let dir = std::env::temp_dir().join("llung-pathfield-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("note.txt"), "hi").unwrap();
        let text = format!("{}/note", dir.display());
        let c = for_path_field(&text).unwrap();
        assert_eq!(c.items[0].label, "note.txt");
        assert_eq!(c.items[0].replace, format!("{}/note.txt", dir.display()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn selection_wraps() {
        let mut c = for_input("/agent spawn scout backend=", 27, "").unwrap();
        let first = c.selected;
        c.move_selection(-1);
        assert_eq!(c.selected, c.items.len() - 1);
        c.move_selection(1);
        assert_eq!(c.selected, first);
    }

    #[test]
    fn fuzzy_prefers_prefix_and_rejects_garbage() {
        assert!(fuzzy("lemonade", "lem") > fuzzy("lemonade", "ade"));
        assert!(fuzzy("kimi-k2-thinking", "kthink").is_some());
        assert!(fuzzy("abc", "zzz").is_none());
        assert!(fuzzy("anything", "").is_some());
    }
}
