// crates/agent/src/tools/host/glob.rs
//
// In-process glob: `ignore::WalkBuilder` (the engine behind ripgrep)
// handles .gitignore/.git_exclude/hidden-file rules natively; `globset`
// matches the pattern. No external processes.

use serde::Deserialize;
use std::path::PathBuf;

/// Hard cap on returned paths — the model's context, not the disk, is
/// the limiting resource.
const MAX_RESULTS: usize = 1000;

#[derive(Deserialize)]
struct GlobArgs {
    /// Glob pattern, e.g. "**/*.rs" (any depth) or "*.rs".
    pattern: String,
    /// Root directory to search (default: process cwd).
    #[serde(default)]
    path: Option<String>,
}

pub fn run(arguments_json: &str) -> Result<String, String> {
    let args: GlobArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("glob: arguments must be JSON with a 'pattern' string: {e}"))?;
    if args.pattern.trim().is_empty() {
        return Err("glob: 'pattern' must not be empty".into());
    }
    let root = args.path.as_deref().unwrap_or(".");
    let root = shellexpand::tilde(root).to_string();
    // Absolute root, so every result is an absolute path the model can
    // hand straight to another tool. (Relative results were relative to
    // `path`, and the model re-used them against the workspace instead
    // — "data/flights.csv" under ~/Code became a miss under the repo.)
    let root_path = std::path::absolute(PathBuf::from(root.as_ref() as &str))
        .map_err(|e| format!("glob: bad path '{root}': {e}"))?;
    if !root_path.is_dir() {
        return Err(format!("glob: '{}' is not a directory", root_path.display()));
    }
    let matcher = globset::GlobBuilder::new(&args.pattern)
        .build()
        .map_err(|e| format!("glob: bad pattern '{}': {e}", args.pattern))?
        .compile_matcher();

    let mut out = String::new();
    let mut count = 0usize;
    let mut truncated = false;
    // git_ignore/git_exclude/hidden all default on: results match what
    // ripgrep would see. `require_git(false)`: ripgrep only honors
    // .gitignore inside a git repo, but an agent's workspace may not be
    // one — the rules should still apply.
    let walker = ignore::WalkBuilder::new(&root_path)
        .require_git(false)
        .build();
    for entry in walker {
        let entry = entry.map_err(|e| format!("glob: {}", e))?;
        let Some(file_type) = entry.file_type() else { continue };
        if !file_type.is_file() {
            continue;
        }
        let rel = entry.path().strip_prefix(&root_path).unwrap_or(entry.path());
        if matcher.is_match(rel) {
            count += 1;
            if count > MAX_RESULTS {
                truncated = true;
                break;
            }
            out.push_str(&entry.path().to_string_lossy());
            out.push('\n');
        }
    }
    if count == 0 {
        out.push_str("(no matches)\n");
    }
    if truncated {
        out.push_str(&format!("… truncated at {MAX_RESULTS} results\n"));
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
                "llung-glob-{tag}-{}-{}",
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

    fn glob_in(root: &std::path::Path, pattern: &str) -> Result<String, String> {
        run(&serde_json::json!({
            "pattern": pattern,
            "path": root.to_string_lossy(),
        })
        .to_string())
    }

    #[test]
    fn finds_nested_files_matching_pattern() {
        let dir = TempDir::new("nested");
        dir.file("src/main.rs", "fn main() {}");
        dir.file("src/lib/util.rs", "pub fn f() {}");
        dir.file("src/lib/util.py", "def f(): pass");
        dir.file("README.md", "# hi");

        let out = glob_in(&dir.0, "**/*.rs").unwrap();
        assert!(out.contains("src/main.rs"), "got: {out}");
        // Absolute paths: usable verbatim by the next tool call,
        // whatever `path` the search was rooted at.
        for line in out.lines() {
            assert!(std::path::Path::new(line).is_absolute(), "relative result: {line}");
            assert!(std::path::Path::new(line).is_file(), "not a real file: {line}");
        }
        assert!(out.contains("src/lib/util.rs"), "got: {out}");
        assert!(!out.contains("util.py"), "got: {out}");
        assert!(!out.contains("README.md"), "got: {out}");
    }

    #[test]
    fn respects_gitignore_rules() {
        let dir = TempDir::new("gitignore");
        dir.file("src/a.rs", "");
        dir.file("target/b.rs", "");
        dir.file(".gitignore", "target/\n");

        let out = glob_in(&dir.0, "**/*.rs").unwrap();
        assert!(out.contains("src/a.rs"), "got: {out}");
        assert!(!out.contains("target/b.rs"), "gitignored file must not appear: {out}");
    }

    #[test]
    fn reports_no_matches_and_bad_roots() {
        let dir = TempDir::new("empty");
        dir.file("a.txt", "");
        let out = glob_in(&dir.0, "**/*.rs").unwrap();
        assert!(out.contains("(no matches)"), "got: {out}");
        assert!(glob_in(std::path::Path::new("/no/such/dir"), "**/*").is_err());
        assert!(run(&serde_json::json!({"pattern": "["}).to_string()).is_err());
    }

    #[test]
    fn truncates_at_the_result_cap() {
        let dir = TempDir::new("cap");
        for i in 0..(MAX_RESULTS + 10) {
            dir.file(&format!("f{i:05}.txt"), "");
        }
        let out = glob_in(&dir.0, "**/*.txt").unwrap();
        assert!(out.contains(&format!("truncated at {MAX_RESULTS}")), "got: {out}");
        // Walk order is filesystem-dependent: assert the SHAPE (exactly
        // MAX_RESULTS distinct results + the truncation note), not which
        // particular files made the cut.
        let result_lines = out
            .lines()
            .filter(|l| !l.starts_with('…'))
            .collect::<Vec<_>>();
        assert_eq!(result_lines.len(), MAX_RESULTS, "got: {out}");
        assert_eq!(
            result_lines.iter().collect::<std::collections::BTreeSet<_>>().len(),
            MAX_RESULTS,
            "results must be distinct: {out}"
        );
    }
}
