// crates/agent/src/tools/host/grep.rs
//
// In-process regex search: `grep-regex` compiles the pattern,
// `grep-searcher` does the line search (binary detection, memory maps),
// `ignore::WalkBuilder` supplies the file list with .gitignore rules.
// No external processes.

use grep_regex::RegexMatcher;
use grep_searcher::{Searcher, Sink, SinkMatch};
use serde::Deserialize;
use std::path::PathBuf;

/// Caps keep a runaway search (huge tree, match-everywhere regex) from
/// blowing up the model's context.
const MAX_MATCHES: usize = 500;
const MAX_OUTPUT_BYTES: usize = 256 * 1024;

#[derive(Deserialize)]
struct GrepArgs {
    /// Regex pattern (Rust regex syntax).
    pattern: String,
    /// File or directory to search (default: process cwd).
    #[serde(default)]
    path: Option<String>,
    /// Optional glob filter on file names, e.g. "*.rs" (gitignore still applies).
    #[serde(default)]
    include: Option<String>,
}

/// Collects matches for ONE file: `path_prefix` is baked in so each
/// output line is `path:line:text` without the sink needing the path.
struct CollectSink<'p> {
    path_prefix: &'p str,
    out: &'p mut String,
    matches: &'p mut usize,
    capped: &'p mut bool,
    /// (line number) of the last emitted line — a line holding several
    /// matches would otherwise print once per match.
    last_line: Option<u64>,
}

impl Sink for CollectSink<'_> {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _searcher: &Searcher,
        mat: &SinkMatch<'_>,
    ) -> Result<bool, Self::Error> {
        if *self.matches >= MAX_MATCHES || self.out.len() >= MAX_OUTPUT_BYTES {
            *self.capped = true;
            return Ok(false); // stop searching this file
        }
        let Some(line_no) = mat.line_number() else {
            return Ok(true);
        };
        if self.last_line == Some(line_no) {
            return Ok(true); // same line, second match — already printed
        }
        self.last_line = Some(line_no);
        let text = String::from_utf8_lossy(mat.bytes());
        let text = text.trim_end_matches(['\n', '\r']);
        *self.matches += 1;
        self.out.push_str(&format!("{}:{}:{}\n", self.path_prefix, line_no, text));
        Ok(true)
    }
}

pub fn run(arguments_json: &str) -> Result<String, String> {
    let args: GrepArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("grep: arguments must be JSON with a 'pattern' string: {e}"))?;
    if args.pattern.is_empty() {
        return Err("grep: 'pattern' must not be empty".into());
    }
    let matcher = RegexMatcher::new(&args.pattern)
        .map_err(|e| format!("grep: bad pattern '{}': {e}", args.pattern))?;

    let root = args.path.as_deref().unwrap_or(".");
    let root = shellexpand::tilde(root).to_string();
    // Absolute root: result lines carry absolute paths (see glob.rs).
    let root_path = std::path::absolute(PathBuf::from(root.as_ref() as &str))
        .map_err(|e| format!("grep: bad path '{root}': {e}"))?;
    if !root_path.exists() {
        return Err(format!("grep: '{}' does not exist", root_path.display()));
    }

    // Files to search: the file itself, or a gitignore-aware walk with
    // an optional include-glob filter.
    let mut files: Vec<PathBuf> = Vec::new();
    if root_path.is_file() {
        files.push(root_path);
    } else {
        // require_git(false): honor .gitignore even when the workspace
        // is not a git repository (same rationale as glob.rs).
        let mut builder = ignore::WalkBuilder::new(&root_path);
        builder.require_git(false);
        if let Some(include) = &args.include {
            let mut overrides = ignore::overrides::OverrideBuilder::new(&root_path);
            overrides
                .add(include)
                .map_err(|e| format!("grep: bad include glob '{include}': {e}"))?;
            builder.overrides(
                overrides
                    .build()
                    .map_err(|e| format!("grep: bad include glob '{include}': {e}"))?,
            );
        }
        for entry in builder.build() {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => return Err(format!("grep: {}", e)),
            };
            if entry.file_type().is_some_and(|t| t.is_file()) {
                files.push(entry.into_path());
            }
        }
    }

    let mut out = String::new();
    let mut matches = 0usize;
    let mut capped = false;
    let mut searcher = grep_searcher::SearcherBuilder::new()
        .line_number(true)
        .build();
    for file in &files {
        if capped {
            break;
        }
        let display = file.display().to_string();
        let mut sink = CollectSink {
            path_prefix: &display,
            out: &mut out,
            matches: &mut matches,
            capped: &mut capped,
            last_line: None,
        };
        searcher
            .search_path(&matcher, file, &mut sink)
            .map_err(|e| format!("grep: searching '{}': {e}", file.display()))?;
    }

    if matches == 0 {
        out.push_str("(no matches)\n");
    }
    if capped {
        out.push_str(&format!(
            "… truncated (cap: {MAX_MATCHES} matches / {} bytes output)\n",
            MAX_OUTPUT_BYTES
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "llung-grep-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn file(&self, rel: &str, content: &str) {
            let path = self.0.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, content).unwrap();
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    fn grep_in(root: &std::path::Path, pattern: &str) -> Result<String, String> {
        run(&serde_json::json!({
            "pattern": pattern,
            "path": root.to_string_lossy(),
        })
        .to_string())
    }

    #[test]
    fn finds_matches_with_path_and_line_numbers() {
        let dir = TempDir::new("basic");
        dir.file("src/main.rs", "fn main() {\n    let needle = 1;\n}\n");
        dir.file("src/other.rs", "let needle = 2;\n");
        let out = grep_in(&dir.0, "needle").unwrap();
        assert!(out.contains("main.rs:2:    let needle = 1;"), "got: {out}");
        assert!(out.contains("other.rs:1:let needle = 2;"), "got: {out}");
    }

    #[test]
    fn include_glob_filters_files_and_gitignore_excludes() {
        let dir = TempDir::new("filters");
        dir.file("src/a.rs", "haystack MARKER\n");
        dir.file("src/b.py", "haystack MARKER\n");
        dir.file("target/c.rs", "MARKER\n");
        dir.file(".gitignore", "target/\n");

        let out = run(&serde_json::json!({
            "pattern": "MARKER",
            "path": dir.0.to_string_lossy(),
            "include": "*.rs",
        })
        .to_string())
        .unwrap();
        assert!(out.contains("a.rs:1"), "got: {out}");
        assert!(!out.contains("b.py"), "got: {out}");
        assert!(!out.contains("target/c.rs"), "gitignored: {out}");
    }

    #[test]
    fn single_file_path_and_regex_syntax() {
        let dir = TempDir::new("onefile");
        dir.file("data.txt", "foo1\nbar\nfoo22\n");
        let out = run(&serde_json::json!({
            "pattern": "^foo\\d+$",
            "path": dir.0.join("data.txt").to_string_lossy(),
        })
        .to_string())
        .unwrap();
        assert!(out.contains("data.txt:1:foo1"), "got: {out}");
        assert!(out.contains("data.txt:3:foo22"), "got: {out}");
        assert!(!out.contains("bar"), "got: {out}");
    }

    #[test]
    fn no_matches_bad_pattern_bad_path() {
        let dir = TempDir::new("errors");
        dir.file("a.txt", "hello\n");
        assert!(grep_in(&dir.0, "zzz").unwrap().contains("(no matches)"));
        assert!(grep_in(&dir.0, "([").is_err());
        assert!(run(&serde_json::json!({
            "pattern": "x",
            "path": "/no/such/place",
        })
        .to_string())
        .is_err());
    }

    #[test]
    fn caps_output_on_match_everywhere() {
        let dir = TempDir::new("cap");
        let mut content = String::new();
        for i in 0..2000 {
            content.push_str(&format!("line {i} match\n"));
        }
        dir.file("big.txt", &content);
        let out = grep_in(&dir.0, "match").unwrap();
        assert!(out.contains("truncated"), "got: {out}");
    }
}
