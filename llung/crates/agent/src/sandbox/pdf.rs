// crates/agent/src/sandbox/pdf.rs
//
// The `pdf_read` tool: PDF -> Markdown via the `llung-pdf` helper
// (crates/llung-pdf: pdf-inspector native extraction, PDFium + PP-OCRv6
// OCR for scanned pages). The helper runs INSIDE bubblewrap — no
// network, the one PDF bound read-only, a memory and time limit —
// because PDFium and ONNX Runtime are C++ parsing untrusted input, and
// the agent host process holds API keys and the grant-signing key.
//
// Tier: FsRead (no prompt). It reads one file the agent may already
// read, writes nothing, and cannot reach the network.
//
// Long documents page on the helper's `<!-- page N -->` markers: the
// output is cut at a page boundary near `max_chars`, with a note telling
// the model which `pages` to ask for next.

use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::sandbox::bwrap::Sandbox;

const DEFAULT_MAX_CHARS: usize = 12_000;
const MAX_MAX_CHARS: usize = 100_000;
/// OCR of a long scanned document is slow; native extraction is ~0.1 s.
const TIMEOUT: Duration = Duration::from_secs(300);
/// PDFium page bitmaps + two ONNX models.
const MEMORY: &str = "3072M";
/// Explicit helper binary (dev builds); otherwise `llung-pdf` on PATH.
const BIN_ENV: &str = "LLUNG_PDF_BIN";

#[derive(Debug, Deserialize)]
struct PdfArgs {
    /// PDF path (workspace-relative, or absolute under a readable root).
    path: String,
    /// 1-indexed page selection, e.g. "1-5" or "2,4,7-9".
    #[serde(default)]
    pages: Option<String>,
    /// auto (default: OCR only pages without usable text) | off | force.
    #[serde(default)]
    ocr: Option<String>,
    /// Output cap in characters (default 12 000, max 100 000).
    #[serde(default)]
    max_chars: Option<usize>,
}

fn parse_args(arguments_json: &str) -> Result<PdfArgs, String> {
    let args: PdfArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("pdf_read: arguments must be JSON with a 'path' string: {e}"))?;
    if args.path.trim().is_empty() {
        return Err("pdf_read: 'path' must not be empty".into());
    }
    if let Some(mode) = args.ocr.as_deref().filter(|m| !matches!(*m, "auto" | "off" | "force")) {
        return Err(format!("pdf_read: ocr must be auto, off or force (got '{mode}')"));
    }
    if let Some(pages) = &args.pages {
        // The helper validates ranges; reject anything that isn't a page
        // list before it becomes an argv element.
        if pages.trim().is_empty()
            || !pages.chars().all(|c| c.is_ascii_digit() || matches!(c, ',' | '-' | ' '))
        {
            return Err(format!("pdf_read: pages must look like \"1-5\" or \"2,4,7-9\" (got '{pages}')"));
        }
    }
    Ok(args)
}

/// The helper binary: `$LLUNG_PDF_BIN`, else `llung-pdf` on PATH.
pub fn helper() -> Option<PathBuf> {
    match std::env::var_os(BIN_ENV).filter(|v| !v.is_empty()) {
        Some(p) => Some(PathBuf::from(p)).filter(|p| p.is_file()),
        None => crate::sandbox::closure::which("llung-pdf"),
    }
}

/// PDF magic within the first KiB (some files carry a preamble).
fn looks_like_pdf(path: &Path) -> bool {
    use std::io::Read;
    let mut head = [0u8; 1024];
    let n = std::fs::File::open(path)
        .and_then(|mut f| f.read(&mut head))
        .unwrap_or(0);
    head[..n].windows(5).any(|w| w == b"%PDF-")
}

pub async fn run(arguments_json: &str, places: &crate::places::Places) -> Result<String, String> {
    let args = parse_args(arguments_json)?;
    let file = places.readable(&args.path)?;
    if !file.is_file() {
        return Err(format!("pdf_read: '{}' is not a file", args.path));
    }
    if !looks_like_pdf(&file) {
        return Err(format!("pdf_read: '{}' is not a PDF (no %PDF- header)", args.path));
    }
    let bin = helper().ok_or(
        "pdf_read: the llung-pdf helper is not installed (flake package llung-pdf)",
    )?;
    let bin = std::fs::canonicalize(&bin).unwrap_or(bin);

    let mut argv = vec![
        file.to_string_lossy().into_owned(),
        "--ocr".into(),
        args.ocr.clone().unwrap_or_else(|| "auto".into()),
    ];
    if let Some(pages) = &args.pages {
        argv.push("--pages".into());
        argv.push(pages.replace(' ', ""));
    }

    let (stdout, stderr, code, note) = if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
        let (o, e, c) = run_unsandboxed(&bin, &argv).await?;
        (o, e, c, Some("[sandbox: OFF — LLUNG_SANDBOX=off, llung-pdf ran unsandboxed]\n"))
    } else {
        let mut sb = Sandbox::baseline(bin.to_string_lossy().into_owned(), argv);
        // The helper's store closure: the binary, its Nix wrapper's
        // PDFium / ONNX Runtime libraries and the pinned OCR models.
        sb.bind_binary_at(&bin).map_err(|e| {
            format!("pdf_read: cannot sandbox {} ({e}) — use the Nix-built helper", bin.display())
        })?;
        sb.bind_ro(&file, &file);
        sb.memory_max(MEMORY);
        let out = sb.run(TIMEOUT).await?;
        if out.timed_out {
            return Err(format!("pdf_read: timed out after {}s", TIMEOUT.as_secs()));
        }
        (out.stdout, out.stderr, out.exit_code, None)
    };
    if code != Some(0) {
        let err = String::from_utf8_lossy(&stderr);
        return Err(format!("pdf_read: {}", err.trim()));
    }
    let text = String::from_utf8_lossy(&stdout);
    let max_chars = args.max_chars.unwrap_or(DEFAULT_MAX_CHARS).clamp(1_000, MAX_MAX_CHARS);
    let mut out = String::new();
    if let Some(note) = note {
        out.push_str(note);
    }
    out.push_str(&page_cut(&text, max_chars));
    Ok(out)
}

async fn run_unsandboxed(bin: &Path, argv: &[String]) -> Result<(Vec<u8>, Vec<u8>, Option<i32>), String> {
    let child = tokio::process::Command::new(bin)
        .args(argv)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .output();
    match tokio::time::timeout(TIMEOUT, child).await {
        Ok(Ok(out)) => Ok((out.stdout, out.stderr, out.status.code())),
        Ok(Err(e)) => Err(format!("pdf_read: cannot run {}: {e}", bin.display())),
        Err(_) => Err(format!("pdf_read: timed out after {}s", TIMEOUT.as_secs())),
    }
}

/// Cut `text` to about `max_chars` at a `<!-- page N` boundary, and say
/// which pages to request next. A single page larger than the cap is
/// cut mid-page (with the same note).
fn page_cut(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    const MARK: &str = "\n<!-- page ";
    let byte_cap = text
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    let starts: Vec<usize> = text.match_indices(MARK).map(|(i, _)| i).collect();
    let page_at = |i: usize| -> Option<u32> {
        text[i + MARK.len()..]
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .and_then(|n| n.parse().ok())
    };
    // Last page boundary under the cap, but never before the first page.
    let cut = starts
        .iter()
        .copied()
        .filter(|&i| Some(i) != starts.first().copied())
        .rfind(|&i| i <= byte_cap);
    let (shown, next_page) = match cut {
        Some(i) => (&text[..i], page_at(i)),
        None => {
            // One page exceeds the cap: hard cut, resume at the next page.
            let mut end = byte_cap;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            let next = starts.iter().copied().find(|&i| i > end).and_then(page_at);
            (&text[..end], next)
        }
    };
    let last_page = starts.last().copied().and_then(page_at);
    let mut out = shown.trim_end().to_string();
    out.push_str("\n\n[… output cut at ~");
    out.push_str(&max_chars.to_string());
    out.push_str(" chars");
    match (next_page, last_page) {
        (Some(next), Some(last)) if next <= last => out.push_str(&format!(
            " — call pdf_read again with pages=\"{next}-{last}\" (or a narrower range) to continue]"
        )),
        _ => out.push_str(" — pass a larger max_chars or a page range to see the rest]"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_are_validated() {
        assert!(parse_args(r#"{"path":"a.pdf"}"#).is_ok());
        assert!(parse_args(r#"{"path":" "}"#).is_err());
        assert!(parse_args(r#"{"path":"a.pdf","ocr":"maybe"}"#).is_err());
        assert!(parse_args(r#"{"path":"a.pdf","pages":"1-3,5"}"#).is_ok());
        assert!(parse_args(r#"{"path":"a.pdf","pages":"1; rm -rf /"}"#).is_err());
        assert!(parse_args(r#"{"path":"a.pdf","pages":"--ocr"}"#).is_err());
    }

    fn doc(pages: u32, per_page: usize) -> String {
        let mut s = format!("[pdf: x.pdf · {pages} pages · showing all · native text]\n");
        for p in 1..=pages {
            s.push_str(&format!("\n<!-- page {p} -->\n{}\n", "w".repeat(per_page)));
        }
        s
    }

    #[test]
    fn long_output_is_cut_on_a_page_boundary_with_a_resume_hint() {
        let text = doc(10, 1_000);
        let cut = page_cut(&text, 3_500);
        assert!(cut.contains("<!-- page 3 -->"));
        assert!(!cut.contains("<!-- page 4 -->"), "cut before the page that overflows");
        assert!(cut.contains(r#"pages="4-10""#), "{}", &cut[cut.len() - 120..]);
        // Short output passes through untouched.
        assert_eq!(page_cut(&doc(2, 10), 3_500), doc(2, 10));
    }

    #[test]
    fn a_single_huge_page_is_hard_cut() {
        let text = doc(2, 20_000);
        let cut = page_cut(&text, 5_000);
        assert!(cut.chars().count() < 5_300);
        assert!(cut.contains(r#"pages="2-2""#), "{}", &cut[cut.len() - 120..]);
    }

    #[test]
    fn non_pdfs_are_rejected_before_any_process_runs() {
        let dir = std::env::temp_dir().join(format!("llung-pdf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("fake.pdf");
        std::fs::write(&fake, "just text").unwrap();
        assert!(!looks_like_pdf(&fake));
        std::fs::write(&fake, "%PDF-1.7\n...").unwrap();
        assert!(looks_like_pdf(&fake));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// End-to-end through the real helper, when one is built: native
    /// text out of a generated PDF. Runs sandboxed when the helper is a
    /// Nix store binary and bubblewrap is present, else unsandboxed via
    /// LLUNG_PDF_BIN (set by the test from cargo's target dir).
    #[tokio::test]
    async fn reads_a_pdf_through_the_helper() {
        let target_bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/llung-pdf");
        let in_store = helper().is_some_and(|p| {
            std::fs::canonicalize(p).is_ok_and(|p| p.starts_with("/nix/store"))
        });
        if !in_store && !target_bin.is_file() {
            eprintln!("no llung-pdf helper (cargo build -p llung-pdf); skipping");
            return;
        }
        let ws = std::env::temp_dir().join(format!("llung-pdf-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("hello.pdf"), minimal_pdf("Hello sandboxed PDF")).unwrap();
        let places = crate::places::Places::fixed(&ws, ws.join("runs"), vec![]);
        if !in_store {
            // SAFETY: test-local env; only this test reads these vars.
            unsafe {
                std::env::set_var(BIN_ENV, &target_bin);
                std::env::set_var("LLUNG_SANDBOX", "off");
            }
        }
        let out = run(r#"{"path":"hello.pdf","ocr":"off"}"#, &places).await;
        if !in_store {
            unsafe {
                std::env::remove_var(BIN_ENV);
                std::env::remove_var("LLUNG_SANDBOX");
            }
        }
        let out = out.expect("pdf_read");
        assert!(out.contains("Hello sandboxed PDF"), "{out}");
        assert!(out.contains("[pdf: hello.pdf · 1 pages"), "{out}");
        std::fs::remove_dir_all(&ws).ok();
    }

    /// A one-page PDF with one line of Helvetica text (valid xref).
    fn minimal_pdf(text: &str) -> Vec<u8> {
        let content = format!("BT /F1 24 Tf 72 720 Td ({text}) Tj ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
             /Resources << /Font << /F1 5 0 R >> >> >>"
                .to_string(),
            format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
        for off in offsets {
            out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }
}
#[cfg(test)]
mod live {
    /// Sandboxed OCR through the Nix-built helper. Manual: put an
    /// image-only PDF with the text "0.42" at $LLUNG_OCR_FIXTURE_DIR/
    /// scanned.pdf, then run with --ignored (helper on PATH or
    /// LLUNG_PDF_BIN=<store path>; needs bubblewrap).
    #[tokio::test]
    #[ignore]
    async fn sandboxed_ocr_of_a_scanned_pdf() {
        let Some(ws) = std::env::var_os("LLUNG_OCR_FIXTURE_DIR").map(std::path::PathBuf::from) else {
            eprintln!("LLUNG_OCR_FIXTURE_DIR not set; skipping");
            return;
        };
        let places = crate::places::Places::fixed(&ws, ws.join("runs"), vec![]);
        let out = super::run(r#"{"path":"scanned.pdf"}"#, &places).await.expect("pdf_read");
        println!("{out}");
        assert!(out.contains("OCR on pages 1"), "{out}");
        assert!(out.contains("0.42"), "{out}");
    }
}
