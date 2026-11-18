// crates/agent/src/tools/host/edit.rs
//
// Exact search-and-replace file editing. Every `old` string must match
// EXACTLY ONCE in the file (0 matches or 2+ are hard errors, like the
// edit-tool contract this mirrors) — no fuzzy patching. On success the
// file is rewritten and the change is returned as a unified diff made
// with `similar`. No external processes.

use serde::Deserialize;
use similar::TextDiff;

const MAX_FILE_BYTES: usize = 1024 * 1024;

#[derive(Deserialize)]
struct EditArgs {
    /// File to edit.
    path: String,
    /// Ordered list of exact replacements; each `old` must match exactly
    /// once, applied in order.
    edits: Vec<Edit>,
}

#[derive(Deserialize)]
struct Edit {
    old: String,
    new: String,
}

pub fn run(arguments_json: &str) -> Result<String, String> {
    let args: EditArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("edit: arguments must be JSON with 'path' and 'edits' [{{old, new}}]: {e}"))?;
    if args.edits.is_empty() {
        return Err("edit: 'edits' must be a non-empty array of {old, new}".into());
    }
    let expanded = shellexpand::tilde(&args.path).to_string();
    let path = std::path::Path::new(expanded.as_ref() as &str);
    let meta = std::fs::metadata(path)
        .map_err(|e| format!("edit: cannot stat '{}': {e}", args.path))?;
    if meta.len() as usize > MAX_FILE_BYTES {
        return Err(format!(
            "edit: '{}' is {} bytes; the cap is {} — split the edit into a smaller file",
            args.path,
            meta.len(),
            MAX_FILE_BYTES
        ));
    }
    let original = std::fs::read_to_string(path)
        .map_err(|e| format!("edit: cannot read '{}': {e}", args.path))?;

    // Apply edits sequentially onto a working copy. Each edit is
    // re-verified against the CURRENT content so overlapping edits fail
    // loudly instead of silently patching a moved target.
    let mut updated = original.clone();
    for (i, edit) in args.edits.iter().enumerate() {
        let n = i + 1;
        if edit.old.is_empty() {
            return Err(format!("edit #{n}: 'old' must not be empty"));
        }
        let count = updated.matches(&edit.old).count();
        if count == 0 {
            return Err(format!(
                "edit #{n}: no match for the old text ({} chars) — the file may differ from what you last saw; re-read it",
                edit.old.len()
            ));
        }
        if count > 1 {
            return Err(format!(
                "edit #{n}: the old text matches {count} places — it must match exactly once; include more surrounding context",
            ));
        }
        updated = updated.replacen(&edit.old, &edit.new, 1);
    }

    if updated == original {
        return Ok("(no changes — old and new were identical)\n".into());
    }
    std::fs::write(path, &updated)
        .map_err(|e| format!("edit: cannot write '{}': {e}", args.path))?;

    let diff = TextDiff::from_lines(&original, &updated);
    let unified = diff
        .unified_diff()
        .context_radius(3)
        .header(&args.path, &args.path)
        .to_string();
    Ok(unified)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempFile(std::path::PathBuf);
    impl TempFile {
        fn new(tag: &str, content: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "llung-edit-{tag}-{}-{}.txt",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::write(&p, content).unwrap();
            Self(p)
        }
        fn content(&self) -> String {
            std::fs::read_to_string(&self.0).unwrap()
        }
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).ok();
        }
    }

    fn edit_call(path: &std::path::Path, old: &str, new: &str) -> String {
        serde_json::json!({
            "path": path.to_string_lossy(),
            "edits": [{"old": old, "new": new}],
        })
        .to_string()
    }

    #[test]
    fn exact_replace_rewrites_file_and_returns_unified_diff() {
        let f = TempFile::new("basic", "alpha\nbeta\ngamma\n");
        let out = run(&edit_call(&f.0, "beta", "BETA")).unwrap();
        assert_eq!(f.content(), "alpha\nBETA\ngamma\n");
        assert!(out.contains("-beta"), "got: {out}");
        assert!(out.contains("+BETA"), "got: {out}");
        assert!(out.contains("--- "), "got: {out}");
    }

    #[test]
    fn zero_matches_and_multi_matches_are_errors_no_write() {
        let f = TempFile::new("nomatch", "one two\n");
        assert!(run(&edit_call(&f.0, "three", "3")).unwrap_err().contains("no match"));
        assert_eq!(f.content(), "one two\n", "failed edit must not write");

        let f2 = TempFile::new("multi", "same\nsame\n");
        let err = run(&edit_call(&f2.0, "same", "x")).unwrap_err();
        assert!(err.contains("matches 2 places"), "got: {err}");
        assert_eq!(f2.content(), "same\nsame\n", "failed edit must not write");
    }

    #[test]
    fn sequential_edits_apply_in_order() {
        let f = TempFile::new("seq", "a\nb\nc\n");
        let args = serde_json::json!({
            "path": f.0.to_string_lossy(),
            "edits": [
                {"old": "a", "new": "A"},
                {"old": "c", "new": "C"},
            ],
        })
        .to_string();
        run(&args).unwrap();
        assert_eq!(f.content(), "A\nb\nC\n");
    }

    #[test]
    fn second_edit_verifies_against_updated_content() {
        // 'b' appears twice initially; after the first edit replaces the
        // first occurrence's line, the second edit's 'old' is unique.
        let f = TempFile::new("verify", "x b\ny b\n");
        let args = serde_json::json!({
            "path": f.0.to_string_lossy(),
            "edits": [
                {"old": "x b", "new": "x B"},
                {"old": "y b", "new": "y B"},
            ],
        })
        .to_string();
        run(&args).unwrap();
        assert_eq!(f.content(), "x B\ny B\n");
    }

    #[test]
    fn identical_old_and_new_reports_no_change() {
        let f = TempFile::new("noop", "stable\n");
        let out = run(&edit_call(&f.0, "stable", "stable")).unwrap();
        assert!(out.contains("no changes"), "got: {out}");
        assert_eq!(f.content(), "stable\n");
    }
}
