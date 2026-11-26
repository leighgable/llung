// crates/agent/src/web.rs
//
// The `web_fetch` tool (Tier::Net — the operator approves each URL
// unless a standing grant covers it). The HOST fetches, not a sandbox:
// sandboxes stay network-less, and the host can enforce what a sandbox
// with a socket could not:
//
// - http(s) only; at most 5 redirects, each hop re-checked;
// - no private, loopback, link-local or CGNAT targets (the agent must
//   not reach the local model server, the router, or cloud metadata
//   endpoints). DNS is resolved once per hop, checked, and PINNED into
//   that hop's client, so a rebinding DNS answer can't swap in
//   127.0.0.1 between the check and the connect;
// - optional host allowlist (`LLUNG_FETCH_ALLOW=github.com,arxiv.org`);
// - bodies capped at 20 MiB.
//
// What comes back:
// - github.com URLs go through the GitHub API / raw files instead of
//   scraping HTML: a repo gives description, top-level files and the
//   README; blob/ gives the raw file; tree/ lists a directory;
//   issues/N and pull/N give the thread. Optional token: `/key set
//   github` (higher rate limits, private repos).
// - HTML is converted to Markdown (scripts, styles, nav dropped);
// - PDFs are saved under <runs>/downloads/ and read with pdf_read;
// - other text is returned as-is; binaries are saved and described.
// Long results page with `start` (a character offset).
//
// Fetched text is UNTRUSTED input to the model (prompt injection), and
// a URL is a way out (anything the agent puts in it reaches the server).
// That is why this tier prompts with the full URL.

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use reqwest::Url;
use serde::Deserialize;

const DEFAULT_MAX_CHARS: usize = 12_000;
const MAX_MAX_CHARS: usize = 100_000;
const MAX_BODY_BYTES: usize = 20 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const TIMEOUT: Duration = Duration::from_secs(30);
const USER_AGENT: &str = concat!("llung-agent/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Deserialize)]
struct FetchArgs {
    url: String,
    /// Character offset to start from (paging through long results).
    #[serde(default)]
    start: Option<usize>,
    /// Output cap in characters (default 12 000, max 100 000).
    #[serde(default)]
    max_chars: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct DownloadArgs {
    url: String,
    /// Relative path under `<runs>/downloads/` (default: the URL's file name).
    #[serde(default)]
    dest: Option<String>,
    /// Replace an existing file (default: keep it and report it).
    #[serde(default)]
    overwrite: bool,
}

/// web_download's hosts when `LLUNG_DOWNLOAD_ALLOW` is unset: dataset and
/// weight hosts (huggingface.co covers its hf.co CDNs, github.com its
/// githubusercontent.com raw/release hosts). `*` allows any public host.
const DEFAULT_DOWNLOAD_ALLOW: &str = "huggingface.co,github.com";
/// web_download size cap (a guard against a runaway stream, not a quota).
const MAX_DOWNLOAD_BYTES: u64 = 64 << 30;
/// web_download: no total deadline (multi-GB files), but a stalled read fails.
const DOWNLOAD_READ_TIMEOUT: Duration = Duration::from_secs(60);

/// Fetch policy; `run` builds it from the environment, tests directly.
#[derive(Clone, Debug, Default)]
struct Policy {
    /// Hosts that may be fetched (suffix match); empty = any public host.
    allow: Vec<String>,
    /// Permit private/loopback targets (tests with a local stub only).
    allow_private: bool,
    /// Override for api.github.com / raw.githubusercontent.com (tests).
    github_api: Option<String>,
    github_raw: Option<String>,
    github_token: Option<String>,
    /// The env var the allowlist came from (error messages).
    allow_var: &'static str,
}

impl Policy {
    fn from_env() -> Self {
        Self {
            allow: parse_allow(&std::env::var("LLUNG_FETCH_ALLOW").unwrap_or_default()),
            allow_var: "LLUNG_FETCH_ALLOW",
            ..Default::default()
        }
    }

    /// web_download's policy: its OWN allowlist (`LLUNG_DOWNLOAD_ALLOW`,
    /// default [`DEFAULT_DOWNLOAD_ALLOW`]), so a standing web_download
    /// grant stays bounded to dataset/weight hosts.
    fn download_from_env() -> Self {
        let raw = std::env::var("LLUNG_DOWNLOAD_ALLOW")
            .unwrap_or_else(|_| DEFAULT_DOWNLOAD_ALLOW.to_string());
        let allow = parse_allow(&raw);
        Self {
            // `*` = any public host (an empty list means "no allowlist").
            allow: if allow.iter().any(|h| h == "*") { Vec::new() } else { allow },
            allow_var: "LLUNG_DOWNLOAD_ALLOW",
            ..Default::default()
        }
    }

    fn host_allowed(&self, host: &str) -> bool {
        if self.allow.is_empty() {
            return true;
        }
        let host = host.to_ascii_lowercase();
        self.allow.iter().any(|a| {
            host == *a
                || host.ends_with(&format!(".{a}"))
                // Allowing github.com covers the API and raw hosts the
                // GitHub routing below fetches from.
                || (a == "github.com" && host.ends_with("githubusercontent.com"))
                // huggingface.co `resolve/` URLs redirect to *.hf.co
                // (cdn-lfs, cas-bridge.xethub).
                || (a == "huggingface.co" && (host == "hf.co" || host.ends_with(".hf.co")))
        })
    }

    fn api(&self) -> &str {
        self.github_api.as_deref().unwrap_or("https://api.github.com")
    }

    fn raw(&self) -> &str {
        self.github_raw.as_deref().unwrap_or("https://raw.githubusercontent.com")
    }
}

fn parse_allow(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|h| h.trim().trim_start_matches('.').to_ascii_lowercase())
        .filter(|h| !h.is_empty())
        .collect()
}

/// Tier::Net executor.
pub async fn run(arguments_json: &str, places: &crate::places::Places) -> Result<String, String> {
    let mut policy = Policy::from_env();
    policy.github_token = llung_core::secrets::load_api_key("github").await.ok();
    run_with(arguments_json, places, &policy).await
}

async fn run_with(
    arguments_json: &str,
    places: &crate::places::Places,
    policy: &Policy,
) -> Result<String, String> {
    let args: FetchArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("web_fetch: arguments must be JSON with a 'url' string: {e}"))?;
    let url = Url::parse(args.url.trim()).map_err(|e| format!("web_fetch: bad url '{}': {e}", args.url))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("web_fetch: only http(s) URLs (got '{}')", url.scheme()));
    }
    let max = args.max_chars.unwrap_or(DEFAULT_MAX_CHARS).clamp(500, MAX_MAX_CHARS);
    let text = match github_route(&url) {
        Some(route) => github(route, policy).await?,
        None => match generic(url, places, policy, max).await? {
            // PDFs page by page range (pdf_read), not by re-downloading.
            Body::Paged(text) => return Ok(text),
            Body::Text(text) => text,
        },
    };
    Ok(page(&text, args.start.unwrap_or(0), max))
}

// ---------------------------------------------------------------------
// HTTP with SSRF protection
// ---------------------------------------------------------------------

struct Fetched {
    url: Url,
    content_type: String,
    body: Vec<u8>,
}

fn public_v4(ip: std::net::Ipv4Addr) -> bool {
    let o = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_unspecified()
        || ip.is_multicast()
        || o[0] == 0
        || (o[0] == 100 && (o[1] & 0xc0) == 64) // 100.64/10 CGNAT
        || (o[0] == 192 && o[1] == 0 && o[2] == 0) // 192.0.0/24
        || (o[0] == 198 && (o[1] & 0xfe) == 18) // 198.18/15 benchmarking
        || o[0] >= 240) // reserved
}

fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => public_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return public_v4(v4);
            }
            let s = v6.segments();
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (s[0] & 0xfe00) == 0xfc00 // fc00::/7 unique local
                || (s[0] & 0xffc0) == 0xfe80 // fe80::/10 link local
                || (s[0] == 0x2001 && s[1] == 0x0db8) // documentation
                || (s[0] == 0x0064 && s[1] == 0xff9b)) // NAT64 (embeds v4)
        }
    }
}

/// Check one hop and return the address to pin the connection to.
async fn vet(url: &Url, policy: &Policy) -> Result<Option<SocketAddr>, String> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("web_fetch: refusing non-http(s) redirect to {url}"));
    }
    let host = url.host_str().ok_or("web_fetch: URL has no host")?;
    if !policy.host_allowed(host) {
        return Err(format!(
            "web_fetch: host '{host}' is not in {} ({})",
            if policy.allow_var.is_empty() { "the allowlist" } else { policy.allow_var },
            policy.allow.join(", ")
        ));
    }
    let port = url.port_or_known_default().unwrap_or(443);
    let addrs: Vec<SocketAddr> = match url.host() {
        Some(url::Host::Ipv4(ip)) => vec![SocketAddr::new(IpAddr::V4(ip), port)],
        Some(url::Host::Ipv6(ip)) => vec![SocketAddr::new(IpAddr::V6(ip), port)],
        _ => tokio::net::lookup_host((host, port))
            .await
            .map_err(|e| format!("web_fetch: cannot resolve '{host}': {e}"))?
            .collect(),
    };
    if addrs.is_empty() {
        return Err(format!("web_fetch: '{host}' resolved to no address"));
    }
    if let Some(bad) = addrs.iter().find(|a| !policy.allow_private && !is_public(a.ip())) {
        return Err(format!(
            "web_fetch: '{host}' resolves to a private/local address ({}) — refused",
            bad.ip()
        ));
    }
    // Domains get pinned (rebinding-proof); IP literals need no pin.
    Ok(match url.host() {
        Some(url::Host::Domain(_)) => addrs.first().copied(),
        _ => None,
    })
}

async fn get(url: Url, accept: Option<&str>, policy: &Policy) -> Result<Fetched, String> {
    let (url, mut resp) = open(url, accept, policy, false).await?;
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| format!("web_fetch: {url}: {e}"))? {
        if body.len() + chunk.len() > MAX_BODY_BYTES {
            return Err(format!(
                "web_fetch: {url} is larger than {} MiB",
                MAX_BODY_BYTES / (1024 * 1024)
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Fetched { url, content_type, body })
}

/// Follow redirects (each hop vetted and DNS-pinned) to a successful
/// response; the caller reads the body. `long` swaps the 30 s total
/// deadline for a per-read timeout (web_download's multi-GB files).
async fn open(
    url: Url,
    accept: Option<&str>,
    policy: &Policy,
    long: bool,
) -> Result<(Url, reqwest::Response), String> {
    let mut url = url;
    for _ in 0..=MAX_REDIRECTS {
        let pin = vet(&url, policy).await?;
        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .user_agent(USER_AGENT);
        builder = if long {
            builder.read_timeout(DOWNLOAD_READ_TIMEOUT)
        } else {
            builder.timeout(TIMEOUT)
        };
        if let (Some(addr), Some(host)) = (pin, url.host_str()) {
            builder = builder.resolve(host, addr);
        }
        let client = builder.build().map_err(|e| format!("web_fetch: {e}"))?;
        let mut req = client.get(url.clone());
        if let Some(accept) = accept {
            req = req.header(reqwest::header::ACCEPT, accept);
        }
        // The GitHub token only ever goes to the GitHub API host (exact
        // match: never a lookalike, never a redirect target elsewhere).
        let api_host = Url::parse(policy.api()).ok().and_then(|u| u.host_str().map(str::to_owned));
        if let Some(token) = policy.github_token.as_ref().filter(|_| {
            api_host.is_some() && url.host_str() == api_host.as_deref()
        }) {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await.map_err(|e| format!("web_fetch: {url}: {e}"))?;
        let status = resp.status();
        if status.is_redirection() {
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("web_fetch: {status} without Location from {url}"))?;
            url = url
                .join(location)
                .map_err(|e| format!("web_fetch: bad redirect '{location}': {e}"))?;
            continue;
        }
        if !status.is_success() {
            let rate_limited = resp
                .headers()
                .get("x-ratelimit-remaining")
                .is_some_and(|v| v == "0");
            let hint = if rate_limited {
                " (GitHub rate limit — store a token with `/key set github`)"
            } else if status.as_u16() == 404 && url.host_str().is_some_and(|h| h.contains("github")) {
                " (private repo or wrong ref? a token helps for private repos: `/key set github`)"
            } else {
                ""
            };
            return Err(format!("web_fetch: HTTP {status} for {url}{hint}"));
        }
        return Ok((url, resp));
    }
    Err(format!("web_fetch: more than {MAX_REDIRECTS} redirects"))
}

// ---------------------------------------------------------------------
// web_download: a file to disk, for the sandboxes
// ---------------------------------------------------------------------

/// Tier::Net executor for `web_download`: stream one URL to
/// `<runs>/downloads/<dest>` (writable and visible in every sandbox, so
/// network-less sandboxed code — data prep, weight conversion — reads
/// the file). Same SSRF vetting as web_fetch; hosts from
/// `LLUNG_DOWNLOAD_ALLOW`.
pub async fn download(arguments_json: &str, places: &crate::places::Places) -> Result<String, String> {
    download_with(arguments_json, places, &Policy::download_from_env()).await
}

/// Validate `dest`: relative, no `..`, no hidden/odd components.
fn download_dest(dest: Option<&str>, url: &Url) -> Result<std::path::PathBuf, String> {
    let raw = match dest.map(str::trim).filter(|d| !d.is_empty()) {
        Some(d) => d.to_string(),
        None => url
            .path_segments()
            .and_then(|mut s| s.next_back())
            .filter(|s| !s.is_empty())
            .ok_or("web_download: the URL has no file name — pass 'dest'")?
            .to_string(),
    };
    let mut out = std::path::PathBuf::new();
    for part in raw.split('/').filter(|p| !p.is_empty()) {
        let ok = !part.starts_with('.')
            && part.len() <= 128
            && part.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '+' | '='));
        if !ok {
            return Err(format!(
                "web_download: bad dest component '{part}' (relative path under downloads/; \
                 [A-Za-z0-9._-+=], no leading dot, no '..')"
            ));
        }
        out.push(part);
    }
    if out.as_os_str().is_empty() {
        return Err("web_download: empty dest".into());
    }
    Ok(out)
}

async fn download_with(
    arguments_json: &str,
    places: &crate::places::Places,
    policy: &Policy,
) -> Result<String, String> {
    use sha2::Digest;
    use tokio::io::AsyncWriteExt;

    let args: DownloadArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("web_download: arguments must be JSON with a 'url' string: {e}"))?;
    let url = Url::parse(args.url.trim())
        .map_err(|e| format!("web_download: bad url '{}': {e}", args.url))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("web_download: only http(s) URLs (got '{}')", url.scheme()));
    }
    let rel = download_dest(args.dest.as_deref(), &url)?;
    let root = places
        .runs_ready()
        .map_err(|e| format!("web_download: {e}"))?
        .join("downloads");
    let path = root.join(&rel);
    if path.exists() && !args.overwrite {
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        return Ok(format!(
            "Already downloaded: {} ({size} bytes). Pass overwrite=true to fetch again.",
            path.display()
        ));
    }
    let parent = path.parent().unwrap_or(&root);
    std::fs::create_dir_all(parent).map_err(|e| format!("web_download: {}: {e}", parent.display()))?;

    let (final_url, mut resp) = open(url.clone(), None, policy, true).await?;
    let total = resp.content_length();
    if total.is_some_and(|t| t > MAX_DOWNLOAD_BYTES) {
        return Err(format!("web_download: {url} is larger than {} GiB", MAX_DOWNLOAD_BYTES >> 30));
    }
    let part = path.with_file_name(format!(
        "{}.part",
        path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()
    ));
    let result: Result<(u64, [u8; 32]), String> = async {
        let mut file = tokio::fs::File::create(&part)
            .await
            .map_err(|e| format!("web_download: {}: {e}", part.display()))?;
        let mut hasher = sha2::Sha256::new();
        let mut written = 0u64;
        while let Some(chunk) = resp.chunk().await.map_err(|e| format!("web_download: {url}: {e}"))? {
            written += chunk.len() as u64;
            if written > MAX_DOWNLOAD_BYTES {
                return Err(format!("web_download: {url} is larger than {} GiB", MAX_DOWNLOAD_BYTES >> 30));
            }
            hasher.update(&chunk);
            file.write_all(&chunk)
                .await
                .map_err(|e| format!("web_download: {}: {e}", part.display()))?;
        }
        file.flush().await.map_err(|e| format!("web_download: {e}"))?;
        if let Some(t) = total.filter(|&t| t != written) {
            return Err(format!("web_download: {url}: got {written} of {t} bytes (connection cut)"));
        }
        Ok((written, hasher.finalize().into()))
    }
    .await;
    let (written, digest) = match result {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            return Err(e);
        }
    };
    std::fs::rename(&part, &path).map_err(|e| format!("web_download: {}: {e}", path.display()))?;
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    let via = if final_url != url { format!(" (via {})", final_url.host_str().unwrap_or("?")) } else { String::new() };
    Ok(format!(
        "Downloaded {url}{via}\n-> {} ({written} bytes, {:.1} MiB)\nsha256 {hex}\n\
         Sandboxes read it at that path (the runs directory, $LLUNG_RUNS_DIR/downloads/{}).",
        path.display(),
        written as f64 / (1024.0 * 1024.0),
        rel.display(),
    ))
}

// ---------------------------------------------------------------------
// Generic pages
// ---------------------------------------------------------------------

enum Body {
    /// Plain text; `start`/`max_chars` paging applies.
    Text(String),
    /// Already cut to size with its own continuation hint (PDFs).
    Paged(String),
}

async fn generic(
    url: Url,
    places: &crate::places::Places,
    policy: &Policy,
    max_chars: usize,
) -> Result<Body, String> {
    let f = get(url, None, policy).await?;
    let is_pdf = f.content_type.starts_with("application/pdf") || f.body.starts_with(b"%PDF-");
    if is_pdf {
        return pdf(&f, places, max_chars).await.map(Body::Paged);
    }
    if f.content_type.contains("html") || (f.content_type.is_empty() && sniff_html(&f.body)) {
        return Ok(Body::Text(html_to_markdown(&String::from_utf8_lossy(&f.body), &f.url)));
    }
    let textual = f.content_type.starts_with("text/")
        || ["json", "xml", "javascript", "yaml", "toml", "markdown", "csv"]
            .iter()
            .any(|t| f.content_type.contains(t));
    if textual || std::str::from_utf8(&f.body).is_ok() {
        return Ok(Body::Text(format!("Source: {}\n\n{}", f.url, String::from_utf8_lossy(&f.body))));
    }
    let path = save_download(&f, places, "bin")?;
    Ok(Body::Text(format!(
        "Source: {}\nBinary content ({}, {} bytes) saved to {} — not shown.",
        f.url,
        if f.content_type.is_empty() { "unknown type" } else { &f.content_type },
        f.body.len(),
        path.display()
    )))
}

fn sniff_html(body: &[u8]) -> bool {
    let head = String::from_utf8_lossy(&body[..body.len().min(512)]).to_ascii_lowercase();
    head.contains("<html") || head.contains("<!doctype html")
}

fn html_to_markdown(html: &str, url: &Url) -> String {
    let title = html
        .find("<title")
        .and_then(|i| html[i..].find('>').map(|j| i + j + 1))
        .and_then(|start| html[start..].find("</title>").map(|end| &html[start..start + end]))
        .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|t| !t.is_empty());
    let converter = htmd::HtmlToMarkdown::builder()
        .skip_tags(vec![
            "script", "style", "noscript", "svg", "nav", "footer", "iframe", "form", "head",
            "button", "template",
        ])
        .build();
    let md = converter.convert(html).unwrap_or_else(|_| html.to_string());
    let mut out = String::new();
    if let Some(title) = title {
        out.push_str(&format!("# {title}\n\n"));
    }
    out.push_str(&format!("Source: {url}\n\n"));
    // Collapse runs of blank lines left by dropped layout elements.
    let mut blank = 0;
    for line in md.lines() {
        if line.trim().is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

/// Save a fetched body under `<runs>/downloads/`, named after the URL.
fn save_download(f: &Fetched, places: &crate::places::Places, ext: &str) -> Result<std::path::PathBuf, String> {
    let dir = places
        .runs_ready()
        .map_err(|e| format!("web_fetch: cannot save the download: {e}"))?
        .join("downloads");
    std::fs::create_dir_all(&dir).map_err(|e| format!("web_fetch: {}: {e}", dir.display()))?;
    let last = f
        .url
        .path_segments()
        .and_then(|mut s| s.next_back())
        .filter(|s| !s.is_empty())
        .unwrap_or("download");
    let mut stem: String = last
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' })
        .take(80)
        .collect();
    let dotted = format!(".{ext}");
    if !stem.to_ascii_lowercase().ends_with(&dotted) {
        stem.push_str(&dotted);
    }
    let path = dir.join(stem.trim_start_matches('.'));
    std::fs::write(&path, &f.body).map_err(|e| format!("web_fetch: {}: {e}", path.display()))?;
    Ok(path)
}

async fn pdf(f: &Fetched, places: &crate::places::Places, max_chars: usize) -> Result<String, String> {
    let path = save_download(f, places, "pdf")?;
    let saved = format!("Source: {}\nPDF saved to {}", f.url, path.display());
    if crate::sandbox::pdf::helper().is_none() {
        return Ok(format!("{saved} (pdf_read is not installed, so it was not converted)."));
    }
    let args = serde_json::json!({ "path": path.to_string_lossy(), "max_chars": max_chars })
        .to_string();
    match crate::sandbox::pdf::run(&args, places).await {
        Ok(text) => Ok(format!("{saved} — read further pages with pdf_read.\n\n{text}")),
        Err(e) => Ok(format!("{saved}; converting it failed: {e}")),
    }
}

// ---------------------------------------------------------------------
// GitHub
// ---------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Gh {
    Repo { owner: String, repo: String },
    /// `rest` = "<ref>/<path…>"
    Blob { owner: String, repo: String, rest: String },
    Tree { owner: String, repo: String, rest: String },
    Thread { owner: String, repo: String, number: u64 },
}

fn github_route(url: &Url) -> Option<Gh> {
    let host = url.host_str()?.to_ascii_lowercase();
    if host != "github.com" && host != "www.github.com" {
        return None;
    }
    let segs: Vec<&str> = url.path_segments()?.filter(|s| !s.is_empty()).collect();
    let (owner, repo) = (segs.first()?.to_string(), segs.get(1)?.trim_end_matches(".git").to_string());
    let rest = || segs[3..].join("/");
    match segs.get(2).copied() {
        None => Some(Gh::Repo { owner, repo }),
        Some("blob" | "raw") if segs.len() > 4 => Some(Gh::Blob { owner, repo, rest: rest() }),
        Some("tree") if segs.len() > 3 => Some(Gh::Tree { owner, repo, rest: rest() }),
        Some("issues" | "pull") => {
            let number = segs.get(3)?.parse().ok()?;
            Some(Gh::Thread { owner, repo, number })
        }
        _ => None, // releases, actions, wiki…: generic HTML
    }
}

async fn github_json(url: String, policy: &Policy) -> Result<serde_json::Value, String> {
    let url = Url::parse(&url).map_err(|e| format!("web_fetch: {e}"))?;
    let f = get(url, Some("application/vnd.github+json"), policy).await?;
    serde_json::from_slice(&f.body).map_err(|e| format!("web_fetch: GitHub API returned non-JSON: {e}"))
}

fn kb(bytes: u64) -> String {
    if bytes < 1024 { format!("{bytes} B") } else { format!("{:.1} KB", bytes as f64 / 1024.0) }
}

fn listing(entries: &serde_json::Value) -> String {
    let mut items: Vec<(bool, String, u64)> = entries
        .as_array()
        .map(|a| {
            a.iter()
                .map(|e| {
                    (
                        e["type"] == "dir",
                        e["name"].as_str().unwrap_or("?").to_string(),
                        e["size"].as_u64().unwrap_or(0),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    items.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    items
        .into_iter()
        .map(|(dir, name, size)| {
            if dir { format!("- {name}/") } else { format!("- {name} ({})", kb(size)) }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn github(route: Gh, policy: &Policy) -> Result<String, String> {
    let api = policy.api().trim_end_matches('/').to_string();
    match route {
        Gh::Repo { owner, repo } => {
            let info = github_json(format!("{api}/repos/{owner}/{repo}"), policy).await?;
            let branch = info["default_branch"].as_str().unwrap_or("main").to_string();
            let files = github_json(format!("{api}/repos/{owner}/{repo}/contents?ref={branch}"), policy)
                .await
                .map(|v| listing(&v))
                .unwrap_or_else(|e| format!("(listing unavailable: {e})"));
            let readme_url = Url::parse(&format!("{api}/repos/{owner}/{repo}/readme"))
                .map_err(|e| e.to_string())?;
            let readme = get(readme_url, Some("application/vnd.github.raw"), policy)
                .await
                .map(|f| String::from_utf8_lossy(&f.body).into_owned())
                .unwrap_or_else(|_| "(no README)".into());
            let mut facts = vec![format!("★ {}", info["stargazers_count"].as_u64().unwrap_or(0))];
            if let Some(lang) = info["language"].as_str() {
                facts.push(lang.into());
            }
            if let Some(lic) = info["license"]["spdx_id"].as_str() {
                facts.push(lic.into());
            }
            facts.push(format!("default branch {branch}"));
            if let Some(pushed) = info["pushed_at"].as_str() {
                facts.push(format!("last push {}", &pushed[..pushed.len().min(10)]));
            }
            Ok(format!(
                "# {owner}/{repo}\n\n{}\n{}\n\n## Files ({branch})\n{files}\n\n\
                 Browse a directory: web_fetch https://github.com/{owner}/{repo}/tree/{branch}/<dir> · \
                 read a file: https://github.com/{owner}/{repo}/blob/{branch}/<path>\n\n## README\n\n{readme}",
                info["description"].as_str().unwrap_or("(no description)"),
                facts.join(" · "),
            ))
        }
        Gh::Blob { owner, repo, rest } => {
            let raw = policy.raw().trim_end_matches('/');
            let url = Url::parse(&format!("{raw}/{owner}/{repo}/{rest}")).map_err(|e| e.to_string())?;
            let f = get(url, None, policy).await?;
            if std::str::from_utf8(&f.body).is_err() {
                return Ok(format!("{owner}/{repo}: {rest} is binary ({}).", kb(f.body.len() as u64)));
            }
            Ok(format!("# {owner}/{repo}: {rest}\n\n{}", String::from_utf8_lossy(&f.body)))
        }
        Gh::Tree { owner, repo, rest } => {
            let (git_ref, path) = rest.split_once('/').unwrap_or((rest.as_str(), ""));
            let v = github_json(format!("{api}/repos/{owner}/{repo}/contents/{path}?ref={git_ref}"), policy)
                .await?;
            Ok(format!(
                "# {owner}/{repo}: {}/ ({git_ref})\n\n{}\n\nRead a file: \
                 https://github.com/{owner}/{repo}/blob/{git_ref}/{}<name>",
                if path.is_empty() { "." } else { path },
                listing(&v),
                if path.is_empty() { String::new() } else { format!("{path}/") },
            ))
        }
        Gh::Thread { owner, repo, number } => {
            let issue = github_json(format!("{api}/repos/{owner}/{repo}/issues/{number}"), policy).await?;
            let kind = if issue.get("pull_request").is_some() { "Pull request" } else { "Issue" };
            let mut out = format!(
                "# {kind} {owner}/{repo}#{number}: {}\n\n{} · by {} · {} comments\n\n{}\n",
                issue["title"].as_str().unwrap_or(""),
                issue["state"].as_str().unwrap_or("?"),
                issue["user"]["login"].as_str().unwrap_or("?"),
                issue["comments"].as_u64().unwrap_or(0),
                issue["body"].as_str().unwrap_or("(no description)"),
            );
            if issue["comments"].as_u64().unwrap_or(0) > 0 {
                let comments = github_json(
                    format!("{api}/repos/{owner}/{repo}/issues/{number}/comments?per_page=50"),
                    policy,
                )
                .await?;
                for c in comments.as_array().into_iter().flatten() {
                    out.push_str(&format!(
                        "\n---\n**{}** ({}):\n\n{}\n",
                        c["user"]["login"].as_str().unwrap_or("?"),
                        c["created_at"].as_str().map(|d| &d[..d.len().min(10)]).unwrap_or(""),
                        c["body"].as_str().unwrap_or(""),
                    ));
                }
            }
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------
// Paging
// ---------------------------------------------------------------------

/// `max` characters of `text` from char offset `start`, with a note on
/// how to continue.
fn page(text: &str, start: usize, max: usize) -> String {
    let total = text.chars().count();
    if start >= total && total > 0 {
        return format!("[start={start} is past the end ({total} chars)]");
    }
    let shown: String = text.chars().skip(start).take(max).collect();
    let end = start + shown.chars().count();
    let mut out = String::new();
    if start > 0 {
        out.push_str(&format!("[… continuing at char {start} of {total}]\n"));
    }
    out.push_str(&shown);
    if end < total {
        out.push_str(&format!(
            "\n\n[… {} more chars — call web_fetch again with the same url and start={end}]",
            total - end
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_and_special_addresses_are_not_public() {
        for ip in [
            "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1",
            "0.0.0.0", "224.0.0.1", "255.255.255.255", "::1", "fe80::1", "fd00::1",
            "::ffff:127.0.0.1", "64:ff9b::7f00:1",
        ] {
            assert!(!is_public(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["140.82.112.3", "1.1.1.1", "2606:4700:4700::1111"] {
            assert!(is_public(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn github_urls_route_to_the_api() {
        let r = |u: &str| github_route(&Url::parse(u).unwrap());
        assert_eq!(r("https://github.com/a/b"), Some(Gh::Repo { owner: "a".into(), repo: "b".into() }));
        assert_eq!(r("https://github.com/a/b.git/"), Some(Gh::Repo { owner: "a".into(), repo: "b".into() }));
        assert_eq!(
            r("https://github.com/a/b/blob/main/src/lib.rs"),
            Some(Gh::Blob { owner: "a".into(), repo: "b".into(), rest: "main/src/lib.rs".into() })
        );
        assert_eq!(
            r("https://github.com/a/b/tree/dev/docs"),
            Some(Gh::Tree { owner: "a".into(), repo: "b".into(), rest: "dev/docs".into() })
        );
        assert_eq!(
            r("https://github.com/a/b/pull/42"),
            Some(Gh::Thread { owner: "a".into(), repo: "b".into(), number: 42 })
        );
        assert_eq!(r("https://github.com/a/b/releases"), None, "generic HTML");
        assert_eq!(r("https://github.com/a"), None);
        assert_eq!(r("https://example.com/a/b"), None);
    }

    #[test]
    fn allowlist_matches_suffixes_and_github_hosts() {
        let p = Policy { allow: vec!["github.com".into(), "arxiv.org".into()], ..Default::default() };
        assert!(p.host_allowed("github.com"));
        assert!(p.host_allowed("api.github.com"));
        assert!(p.host_allowed("raw.githubusercontent.com"));
        assert!(p.host_allowed("export.arxiv.org"));
        assert!(!p.host_allowed("evil-github.com"));
        assert!(!p.host_allowed("example.com"));
        assert!(Policy::default().host_allowed("anything.org"), "no allowlist: any public host");
    }

    #[test]
    fn html_becomes_markdown_without_scripts() {
        let html = "<html><head><title> Paper  Title </title><script>evil()</script></head>\
                    <body><nav>menu</nav><h1>Results</h1><p>Loss <b>fell</b>.</p>\
                    <script>more()</script></body></html>";
        let md = html_to_markdown(html, &Url::parse("https://x.org/p").unwrap());
        assert!(md.starts_with("# Paper Title\n"), "{md}");
        assert!(md.contains("Source: https://x.org/p"));
        assert!(md.contains("# Results") && md.contains("**fell**"), "{md}");
        assert!(!md.contains("evil") && !md.contains("more()") && !md.contains("menu"), "{md}");
    }

    #[test]
    fn paging_reports_how_to_continue() {
        let text = "a".repeat(100);
        let p = page(&text, 0, 40);
        assert!(p.contains("start=40") && p.contains("60 more chars"), "{p}");
        let p = page(&text, 90, 40);
        assert!(p.starts_with("[… continuing at char 90 of 100]"));
        assert!(!p.contains("more chars"));
        assert!(page(&text, 500, 40).contains("past the end"));
    }

    /// Local HTTP stub: answers each connection from `routes`
    /// (path prefix -> raw response head + body).
    async fn stub(routes: Vec<(&'static str, String)>) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else { return };
                let routes = routes.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let path = req.split_whitespace().nth(1).unwrap_or("/").to_string();
                    let resp = routes
                        .iter()
                        .find(|(p, _)| path.starts_with(p))
                        .map(|(_, r)| r.clone())
                        .unwrap_or_else(|| "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".into());
                    let _ = socket.write_all(resp.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}")
    }

    fn ok(content_type: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn places(tag: &str) -> crate::places::Places {
        let ws = std::env::temp_dir().join(format!("llung-web-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(ws.join("runs")).unwrap();
        crate::places::Places::fixed(&ws, ws.join("runs"), vec![])
    }

    #[tokio::test]
    async fn loopback_targets_are_refused_by_default() {
        let base = stub(vec![("/", ok("text/plain", "secret"))]).await;
        let err = run_with(&serde_json::json!({ "url": base }).to_string(), &places("ssrf"), &Policy::default())
            .await
            .unwrap_err();
        assert!(err.contains("private/local address"), "{err}");
        let err = run_with(r#"{"url":"file:///etc/passwd"}"#, &places("ssrf"), &Policy::default())
            .await
            .unwrap_err();
        assert!(err.contains("only http(s)"), "{err}");
    }

    #[tokio::test]
    async fn redirects_html_and_text_are_handled() {
        let base = stub(vec![
            ("/moved", "HTTP/1.1 301 Moved\r\nlocation: /page\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".into()),
            ("/page", ok("text/html; charset=utf-8", "<title>T</title><p>Hello <i>web</i></p><script>x()</script>")),
            ("/data.json", ok("application/json", r#"{"k": 1}"#)),
        ])
        .await;
        let policy = Policy { allow_private: true, ..Default::default() };
        let pl = places("generic");
        let out = run_with(&serde_json::json!({ "url": format!("{base}/moved") }).to_string(), &pl, &policy)
            .await
            .unwrap();
        assert!(out.contains("# T") && out.contains("Hello *web*") && out.contains("/page"), "{out}");
        assert!(!out.contains("x()"));
        let out = run_with(&serde_json::json!({ "url": format!("{base}/data.json") }).to_string(), &pl, &policy)
            .await
            .unwrap();
        assert!(out.contains(r#"{"k": 1}"#), "{out}");
    }

    #[tokio::test]
    async fn github_repo_view_uses_the_api() {
        let api = stub(vec![
            ("/repos/o/r/readme", ok("text/plain", "# r\nA readme.")),
            (
                "/repos/o/r/contents",
                ok("application/json", r#"[{"name":"src","type":"dir","size":0},{"name":"Cargo.toml","type":"file","size":2048}]"#),
            ),
            (
                "/repos/o/r",
                ok("application/json", r#"{"description":"A crate","default_branch":"main","stargazers_count":7,"language":"Rust","license":{"spdx_id":"MIT"}}"#),
            ),
        ])
        .await;
        let policy = Policy { allow_private: true, github_api: Some(api), ..Default::default() };
        let out = run_with(r#"{"url":"https://github.com/o/r"}"#, &places("gh"), &policy).await.unwrap();
        assert!(out.starts_with("# o/r\n\nA crate"), "{out}");
        assert!(out.contains("★ 7 · Rust · MIT · default branch main"), "{out}");
        assert!(out.contains("- src/\n- Cargo.toml (2.0 KB)"), "dirs first: {out}");
        assert!(out.contains("A readme."));
    }

    #[test]
    fn download_allowlist_covers_hf_cdns_and_star_opens_it() {
        let p = Policy { allow: parse_allow(DEFAULT_DOWNLOAD_ALLOW), ..Default::default() };
        assert!(p.host_allowed("huggingface.co"));
        assert!(p.host_allowed("cdn-lfs.hf.co"));
        assert!(p.host_allowed("cas-bridge.xethub.hf.co"));
        assert!(p.host_allowed("objects.githubusercontent.com"));
        assert!(!p.host_allowed("evilhf.co"));
        assert!(!p.host_allowed("example.com"));
    }

    #[test]
    fn download_dest_is_confined() {
        let u = Url::parse("https://huggingface.co/d/resolve/main/file-1.json.gz?x=1").unwrap();
        assert_eq!(download_dest(None, &u).unwrap(), std::path::PathBuf::from("file-1.json.gz"));
        assert_eq!(
            download_dest(Some("cp/a.gz"), &u).unwrap(),
            std::path::PathBuf::from("cp/a.gz")
        );
        for bad in ["../x", "a/../../x", ".ssh/k", "a b", "x/.", "/"] {
            assert!(download_dest(Some(bad), &u).is_err(), "{bad}");
        }
        assert_eq!(download_dest(Some("/abs/x"), &u).unwrap(), std::path::PathBuf::from("abs/x"));
        let bare = Url::parse("https://huggingface.co/").unwrap();
        assert!(download_dest(None, &bare).is_err());
    }

    #[tokio::test]
    async fn download_streams_to_runs_downloads() {
        let body = "x".repeat(5000);
        let base = stub(vec![
            ("/moved", "HTTP/1.1 302 Found\r\nlocation: /blob.bin\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".into()),
            ("/blob.bin", ok("application/octet-stream", &body)),
        ])
        .await;
        let policy = Policy { allow_private: true, ..Default::default() };
        let pl = places("download");
        let args = serde_json::json!({ "url": format!("{base}/moved"), "dest": "set/a.bin" }).to_string();
        let out = download_with(&args, &pl, &policy).await.unwrap();
        let path = pl.runs().join("downloads/set/a.bin");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), body);
        assert!(out.contains("5000 bytes") && out.contains("sha256 "), "{out}");
        assert!(!pl.runs().join("downloads/set/a.bin.part").exists());
        // Second call keeps the file.
        let out = download_with(&args, &pl, &policy).await.unwrap();
        assert!(out.starts_with("Already downloaded"), "{out}");
        // Loopback is refused without allow_private; hosts outside the list too.
        let err = download_with(&args.replace("set/a", "set/b"), &pl, &Policy::default()).await.unwrap_err();
        assert!(err.contains("private/local"), "{err}");
        let strict = Policy { allow: vec!["huggingface.co".into()], allow_private: true, ..Default::default() };
        let err = download_with(&args.replace("set/a", "set/c"), &pl, &strict).await.unwrap_err();
        assert!(err.contains("not in"), "{err}");
        std::fs::remove_dir_all(pl.runs().parent().unwrap()).ok();
    }

    /// Live network (run with `--ignored`): a real repo, a real file,
    /// a real arXiv PDF through pdf_read.
    #[tokio::test]
    #[ignore]
    async fn live_github_and_arxiv() {
        let pl = places("live");
        let policy = Policy::from_env();
        let repo = run_with(r#"{"url":"https://github.com/firecrawl/pdf-inspector"}"#, &pl, &policy)
            .await
            .unwrap();
        println!("{}\n=====", &repo[..repo.len().min(1200)]);
        assert!(repo.contains("## Files") && repo.contains("## README"));
        let file = run_with(
            r#"{"url":"https://github.com/firecrawl/pdf-inspector/blob/main/Cargo.toml","max_chars":600}"#,
            &pl,
            &policy,
        )
        .await
        .unwrap();
        println!("{file}\n=====");
        assert!(file.contains("[package]"));
        let paper = run_with(r#"{"url":"https://arxiv.org/pdf/1706.03762","max_chars":1500}"#, &pl, &policy)
            .await
            .unwrap();
        println!("{paper}");
        assert!(paper.contains("PDF saved to"));
    }
}

