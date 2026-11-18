//! Markdown-lite for chat messages: just enough structure to make
//! model replies readable — headings, fenced code blocks (drawn as
//! tinted panels), quotes, lists, rules, and inline `code`,
//! **strong**, *emphasis*, [links](url).
//!
//! Deliberately not CommonMark: ambiguous markers stay literal, and a
//! message with no markers parses to a single `Para` so callers can
//! keep their plain-text fast path.

use crate::theme::Role;

#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    /// `#`-prefixed heading; the u8 is the level (1-6), kept for weight.
    Heading(u8, String),
    /// Plain paragraph (may still contain blank-line-separated lines).
    Para(String),
    /// Fenced code block (``` or ~~~); the string is the raw code.
    Code(String),
    /// `> ` quote; the string is the quoted text.
    Quote(String),
    /// `---` / `***` / `___` on their own line.
    Hr,
    /// `- `/`* ` list item; the marker is kept for styling.
    List { marker: String, text: String },
}

/// True when `text` contains any block-level markdown structure or
/// inline markers — callers use this to pick the styled render path.
pub fn has_markdown(text: &str) -> bool {
    blocks(text).len() > 1
        || matches!(blocks(text).first(), Some(Block::Heading(..) | Block::Code(_) | Block::Quote(_) | Block::Hr | Block::List { .. }))
        || text.contains('`')
        || text.contains("**")
        || text.contains("__")
        || text.contains("](")
}

/// Split message text into blocks. Lines are preserved verbatim inside
/// blocks; wrapping happens later, per block.
pub fn blocks(text: &str) -> Vec<Block> {
    let mut out = Vec::new();
    let mut para = String::new();
    let mut code: Option<String> = None;

    let flush_para = |para: &mut String, out: &mut Vec<Block>| {
        let p = std::mem::take(para);
        let trimmed = p.trim_end().to_string();
        if !trimmed.is_empty() {
            out.push(Block::Para(trimmed));
        }
    };

    for line in text.lines() {
        let t = line.trim_end();
        let is_fence = t.trim_start().starts_with("```") || t.trim_start().starts_with("~~~");

        if code.is_some() {
            if is_fence {
                out.push(Block::Code(code.take().unwrap()));
            } else {
                let c = code.as_mut().unwrap();
                c.push_str(t);
                c.push('\n');
            }
            continue;
        }
        if is_fence {
            flush_para(&mut para, &mut out);
            code = Some(String::new());
            continue;
        }

        if t.trim().is_empty() {
            flush_para(&mut para, &mut out);
            continue;
        }

        // Heading: 1-6 '#' followed by a space.
        if let Some(stripped) = strip_heading(t) {
            flush_para(&mut para, &mut out);
            out.push(stripped);
            continue;
        }
        // Horizontal rule: 3+ of -, *, _ (alone, allowing spaces).
        let rule = t.trim();
        if rule.len() >= 3
            && (rule.chars().all(|c| c == '-') || rule.chars().all(|c| c == '*') || rule.chars().all(|c| c == '_'))
        {
            flush_para(&mut para, &mut out);
            out.push(Block::Hr);
            continue;
        }
        // Quote.
        if let Some(q) = t.strip_prefix('>').map(str::trim_start) {
            flush_para(&mut para, &mut out);
            out.push(Block::Quote(q.to_string()));
            continue;
        }
        // List item.
        if let Some((marker, text)) = strip_list_marker(t) {
            flush_para(&mut para, &mut out);
            out.push(Block::List { marker, text });
            continue;
        }
        // Ordinary line: accumulate into the paragraph.
        if !para.is_empty() {
            para.push('\n');
        }
        para.push_str(t);
    }
    if let Some(c) = code {
        // Unclosed fence: still show the code (lenient).
        let c = c.trim_end_matches('\n').to_string();
        if !c.is_empty() {
            out.push(Block::Code(c));
        }
    } else {
        flush_para(&mut para, &mut out);
    }
    out
}

fn strip_heading(line: &str) -> Option<Block> {
    let hashes = line.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) {
        if let Some(rest) = line.get(hashes..).and_then(|r| r.strip_prefix(' ')) {
            return Some(Block::Heading(hashes as u8, rest.trim().to_string()));
        }
    }
    None
}

fn strip_list_marker(line: &str) -> Option<(String, String)> {
    for pat in ["- ", "* ", "+ "] {
        if let Some(rest) = line.strip_prefix(pat) {
            return Some((pat.trim().to_string(), rest.to_string()));
        }
    }
    // Ordered: "1. ", "12. "
    let digits = line.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && digits < 5 {
        if let Some(rest) = line.get(digits..).and_then(|r| r.strip_prefix(". ")) {
            let marker = format!("{}.", &line[..digits]);
            return Some((marker, rest.to_string()));
        }
    }
    None
}

/// Inline markdown for ONE already-wrapped line: `` `code` `` → MdCode,
/// `**strong**`/`__strong__` → MdStrong, `*em*`/`_em_` → MdEm, and
/// `[text](url)` → MdLink text + dimmed URL. Unmatched markers stay
/// literal. Returns styled runs covering the whole line.
pub fn inline(line: &str, base: Role) -> Vec<(String, Role)> {
    let mut runs: Vec<(String, Role)> = Vec::new();
    let mut rest = line;

    fn push(runs: &mut Vec<(String, Role)>, text: &str, role: Role) {
        if text.is_empty() {
            return;
        }
        if let Some(last) = runs.last_mut() {
            if last.1 == role {
                last.0.push_str(text);
                return;
            }
        }
        runs.push((text.to_string(), role));
    }

    while !rest.is_empty() {
        // Find the nearest marker.
        let candidates = [
            rest.find('`').map(|i| (i, Marker::Code)),
            rest.find("**").map(|i| (i, Marker::Strong)),
            rest.find("__").map(|i| (i, Marker::Strong2)),
            rest.find('*').map(|i| (i, Marker::Em)),
            rest.find('_').map(|i| (i, Marker::Em2)),
            rest.find('[').map(|i| (i, Marker::Link)),
        ];
        let Some((at, marker)) = candidates.into_iter().flatten().min_by_key(|(i, _)| *i) else {
            push(&mut runs, rest, base);
            break;
        };
        push(&mut runs, &rest[..at], base);
        let after = &rest[at..];
        match marker {
            Marker::Code => {
                if let Some(end) = after[1..].find('`') {
                    push(&mut runs, &after[1..1 + end], Role::MdCode);
                    rest = &after[1 + end + 1..];
                } else {
                    push(&mut runs, "`", base);
                    rest = &after[1..];
                }
            }
            Marker::Strong | Marker::Strong2 => {
                let pat = if marker == Marker::Strong { "**" } else { "__" };
                if let Some(end) = after[pat.len()..].find(pat) {
                    push(&mut runs, &after[pat.len()..pat.len() + end], Role::MdStrong);
                    rest = &after[pat.len() + end + pat.len()..];
                } else {
                    push(&mut runs, pat, base);
                    rest = &after[pat.len()..];
                }
            }
            Marker::Em | Marker::Em2 => {
                let pat = if marker == Marker::Em { "*" } else { "_" };
                // Emphasis only when the delimiters sit OUTSIDE word
                // characters — snake_case and "5 * 3" stay literal.
                let prev_ok = at == 0 || !rest.as_bytes()[at - 1].is_ascii_alphanumeric();
                if prev_ok {
                    if let Some(end) = after[1..].find(pat) {
                        let inner = &after[1..1 + end];
                        let tail = &after[1 + end + 1..];
                        let next_ok = tail.chars().next().map_or(true, |c| !c.is_alphanumeric());
                        if next_ok && !inner.trim().is_empty() {
                            push(&mut runs, inner, Role::MdEm);
                            rest = tail;
                            continue;
                        }
                    }
                }
                push(&mut runs, pat, base);
                rest = &after[1..];
            }
            Marker::Link => {
                // [text](url)
                if let Some(close) = after.find(']') {
                    let text = &after[1..close];
                    let after_text = &after[close + 1..];
                    if let Some(url_len) = after_text.strip_prefix('(').and_then(|r| r.find(')')) {
                        let url = &after_text[1..1 + url_len];
                        push(&mut runs, text, Role::MdLink);
                        push(&mut runs, &format!(" ({url})"), Role::MdLinkUrl);
                        rest = &after_text[1 + url_len + 1..];
                        continue;
                    }
                }
                push(&mut runs, "[", base);
                rest = &after[1..];
            }
        }
    }
    runs
}

#[derive(Clone, Copy, PartialEq)]
enum Marker {
    Code,
    Strong,
    Strong2,
    Em,
    Em2,
    Link,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_one_para() {
        assert_eq!(blocks("hello world"), vec![Block::Para("hello world".into())]);
        assert!(!has_markdown("just words, no markers"));
        assert!(has_markdown("has `code` inline"));
    }

    #[test]
    fn fenced_code_block() {
        let b = blocks("intro\n```rust\nfn main() {\n    loop {}\n}\n```\noutro\n");
        assert_eq!(b[0], Block::Para("intro".into()));
        assert_eq!(b[1], Block::Code("fn main() {\n    loop {}\n}\n".into()));
        assert_eq!(b[2], Block::Para("outro".into()));
    }

    #[test]
    fn unclosed_fence_still_renders_code() {
        let b = blocks("```\nlet x = 1;\n");
        assert_eq!(b, vec![Block::Code("let x = 1;".into())]);
    }

    #[test]
    fn headings_quotes_lists_rules() {
        let b = blocks("## Plan\n> quoted\n- one\n1. first\n---\n");
        assert_eq!(b[0], Block::Heading(2, "Plan".into()));
        assert_eq!(b[1], Block::Quote("quoted".into()));
        assert_eq!(b[2], Block::List { marker: "-".into(), text: "one".into() });
        assert_eq!(b[3], Block::List { marker: "1.".into(), text: "first".into() });
        assert_eq!(b[4], Block::Hr);
    }

    #[test]
    fn inline_styling() {
        use crate::theme::Role::*;
        let runs = inline("use `cargo test` and **ship it** now", Plain);
        assert!(runs.iter().any(|(t, r)| t == "cargo test" && *r == MdCode));
        assert!(runs.iter().any(|(t, r)| t == "ship it" && *r == MdStrong));

        let runs = inline("see [the docs](https://example.com) here", Plain);
        assert!(runs.iter().any(|(t, r)| t == "the docs" && *r == MdLink));
        assert!(runs.iter().any(|(t, r)| t.contains("https://example.com") && *r == MdLinkUrl));

        // Unmatched markers stay literal.
        let runs = inline("5 * 3 = 15", Plain);
        assert!(runs.iter().all(|(_, r)| *r == Plain));

        // snake_case doesn't become emphasis.
        let runs = inline("my_var_name here", Plain);
        assert!(runs.iter().all(|(_, r)| *r == Plain));
    }
}
