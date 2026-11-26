//! `llung-pdf`: PDF -> Markdown for the agent's `pdf_read` tool.
//!
//! Runs INSIDE the agent sandbox (bubblewrap: no network, the PDF bound
//! read-only), never in the agent host process: PDFium and ONNX Runtime
//! are C++ parsing hostile input. Native text extraction (pdf-inspector,
//! pure Rust) runs first; only pages it flags as scanned/broken are
//! rendered with PDFium and read with PP-OCRv6 (ONNX Runtime).
//!
//!   llung-pdf <file.pdf> [--pages 1,3-5] [--ocr auto|off|force]
//!
//! Runtime configuration (the Nix wrapper sets all three):
//!   PDFIUM_LIB_PATH   libpdfium shared library
//!   ORT_DYLIB_PATH    libonnxruntime shared library
//!   LLUNG_OCR_MODELS  directory with the pinned PP-OCRv6 Small files
//! Without them OCR is skipped with a note; native extraction still works.
//!
//! Output (stdout): a one-line `[pdf: …]` header, then per-page Markdown
//! separated by `<!-- page N -->` markers (the agent side pages long
//! documents on those markers). Exit 1 + stderr on failure.

use std::collections::BTreeSet;
use std::process::ExitCode;

use pdf_inspector::MarkdownProfile;
use pdf_inspector::vision::{
    ModelDownloadPolicy, OcrMode, OcrOptions, OcrPdfOptions, PageContentSource,
    process_pdf_with_ocr_mem,
};

struct Args {
    file: String,
    pages: Option<BTreeSet<u32>>,
    ocr: OcrMode,
}

const USAGE: &str = "usage: llung-pdf <file.pdf> [--pages 1,3-5] [--ocr auto|off|force]";

/// Upper bound on a page range, so `--pages 1-4000000000` can't allocate.
const MAX_PAGE: u32 = 100_000;

fn parse_pages(spec: &str) -> Result<BTreeSet<u32>, String> {
    let mut pages = BTreeSet::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (a, b) = match part.split_once('-') {
            Some((a, b)) => (a.trim(), b.trim()),
            None => (part, part),
        };
        let a: u32 = a.parse().map_err(|_| format!("bad page '{part}'"))?;
        let b: u32 = b.parse().map_err(|_| format!("bad page '{part}'"))?;
        if a == 0 || b < a || b > MAX_PAGE {
            return Err(format!("bad page range '{part}' (pages are 1-indexed)"));
        }
        pages.extend(a..=b);
    }
    if pages.is_empty() {
        return Err(format!("empty page selection '{spec}'"));
    }
    Ok(pages)
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut file = None;
    let mut pages = None;
    let mut ocr = OcrMode::Auto;
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--pages" => pages = Some(parse_pages(it.next().ok_or(USAGE)?)?),
            "--ocr" => {
                ocr = match it.next().map(String::as_str) {
                    Some("auto") => OcrMode::Auto,
                    Some("off") => OcrMode::Off,
                    Some("force") => OcrMode::Force,
                    _ => return Err(USAGE.into()),
                }
            }
            "-h" | "--help" => return Err(USAGE.into()),
            other if file.is_none() && !other.starts_with("--") => file = Some(other.to_string()),
            _ => return Err(USAGE.into()),
        }
    }
    Ok(Args {
        file: file.ok_or(USAGE)?,
        pages,
        ocr,
    })
}

/// Whether the OCR runtime looks present (libraries + models), so an
/// `auto` request on a machine without it degrades to native text.
fn ocr_runtime() -> Result<std::path::PathBuf, String> {
    let lib = |var: &str| match std::env::var_os(var) {
        Some(p) if std::path::Path::new(&p).is_file() => Ok(()),
        Some(p) => Err(format!("{var}={} does not exist", p.to_string_lossy())),
        None => Err(format!("{var} is not set")),
    };
    lib("PDFIUM_LIB_PATH")?;
    lib("ORT_DYLIB_PATH")?;
    match std::env::var_os("LLUNG_OCR_MODELS") {
        Some(dir) if std::path::Path::new(&dir).is_dir() => Ok(dir.into()),
        Some(dir) => Err(format!("LLUNG_OCR_MODELS={} is not a directory", dir.to_string_lossy())),
        None => Err("LLUNG_OCR_MODELS is not set".into()),
    }
}

fn options(args: &Args, mode: OcrMode, models: Option<std::path::PathBuf>) -> OcrPdfOptions {
    let mut ocr = OcrOptions::new()
        .mode(mode)
        .minimum_confidence(0.45)
        // Never touch the network: the sandbox has none anyway, and the
        // models are a pinned, pre-fetched directory.
        .model_downloads(ModelDownloadPolicy::Offline);
    if let Some(dir) = models {
        ocr = ocr.model_directory(dir);
    }
    let mut opts = OcrPdfOptions::new().ocr(ocr);
    // Token-efficient Markdown: this text goes into a model's context.
    opts.markdown.profile = MarkdownProfile::Compact;
    if let Some(pages) = &args.pages {
        opts = opts.page_numbers(pages.iter().copied());
    }
    opts
}

fn page_list(pages: &[u32]) -> String {
    pages.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

fn run(args: &Args) -> Result<String, String> {
    let bytes = std::fs::read(&args.file).map_err(|e| format!("cannot read {}: {e}", args.file))?;
    let mut notes: Vec<String> = Vec::new();

    let (mode, models) = match (args.ocr, ocr_runtime()) {
        (OcrMode::Off, _) => (OcrMode::Off, None),
        (mode, Ok(models)) => (mode, Some(models)),
        (OcrMode::Force, Err(e)) => return Err(format!("OCR forced but unavailable: {e}")),
        (_, Err(e)) => {
            notes.push(format!("OCR unavailable ({e}); native text only"));
            (OcrMode::Off, None)
        }
    };
    let result = match process_pdf_with_ocr_mem(&bytes, options(args, mode, models)) {
        Ok(r) => r,
        // OCR failed (bad runtime, model mismatch): native text is still
        // worth returning, with the reason.
        Err(e) if mode != OcrMode::Off => {
            notes.push(format!("OCR failed ({e}); native text only"));
            process_pdf_with_ocr_mem(&bytes, options(args, OcrMode::Off, None))
                .map_err(|e| format!("PDF extraction failed: {e}"))?
        }
        Err(e) => return Err(format!("PDF extraction failed: {e}")),
    };

    if !result.pages_routed_to_ocr.is_empty() {
        notes.push(format!("OCR on pages {}", page_list(&result.pages_routed_to_ocr)));
    }
    let unread: Vec<u32> = result
        .pages_recommended_for_ocr
        .iter()
        .copied()
        .filter(|p| !result.pages_routed_to_ocr.contains(p))
        .collect();
    if !unread.is_empty() {
        notes.push(format!(
            "pages {} look scanned/garbled and were NOT OCR'd — their text may be missing",
            page_list(&unread)
        ));
    }
    // "Hosted recommended" also fires for pages OCR never ran on (they
    // are already reported as unread above): only OCR'd pages count.
    let weak: Vec<u32> = result
        .pages_recommending_hosted
        .iter()
        .copied()
        .filter(|p| result.pages_routed_to_ocr.contains(p))
        .collect();
    if !weak.is_empty() {
        notes.push(format!("low OCR confidence on pages {}", page_list(&weak)));
    }
    if !result.pages_with_tables.is_empty() {
        notes.push(format!("tables on pages {}", page_list(&result.pages_with_tables)));
    }

    let shown: Vec<u32> = result.pages.iter().map(|p| p.page_number).collect();
    let mut out = format!(
        "[pdf: {} · {} pages · showing {} · {}]\n",
        std::path::Path::new(&args.file)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| args.file.clone()),
        result.page_count,
        if shown.len() as u32 == result.page_count {
            "all".to_string()
        } else {
            format!("pages {}", page_list(&shown))
        },
        if notes.is_empty() { "native text".to_string() } else { notes.join(" · ") },
    );
    for page in &result.pages {
        let source = match page.provenance.source {
            PageContentSource::Native => "",
            PageContentSource::Ocr => " (OCR)",
            PageContentSource::Fused => " (text + OCR)",
            #[allow(unreachable_patterns)]
            _ => "",
        };
        out.push_str(&format!("\n<!-- page {}{source} -->\n", page.page_number));
        out.push_str(page.markdown.trim_end());
        out.push('\n');
    }
    Ok(out)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("llung-pdf: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_selection_parses_ranges_and_rejects_nonsense() {
        assert_eq!(parse_pages("1,3-5").unwrap().into_iter().collect::<Vec<_>>(), [1, 3, 4, 5]);
        assert_eq!(parse_pages(" 2 , 2-3 ").unwrap().len(), 2);
        for bad in ["0", "5-3", "x", "", "1-4000000000"] {
            assert!(parse_pages(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn args_default_to_auto_ocr() {
        let a = parse_args(&["f.pdf".into()]).unwrap();
        assert_eq!(a.ocr, OcrMode::Auto);
        assert!(a.pages.is_none());
        let a = parse_args(&["--ocr".into(), "off".into(), "f.pdf".into(), "--pages".into(), "2".into()])
            .unwrap();
        assert_eq!(a.ocr, OcrMode::Off);
        assert!(parse_args(&[]).is_err());
        assert!(parse_args(&["a.pdf".into(), "b.pdf".into()]).is_err());
    }
}
