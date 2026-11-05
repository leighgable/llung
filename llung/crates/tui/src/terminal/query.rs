//! Ask the terminal what it is: default foreground (OSC 10), background
//! (OSC 11), the 16 ANSI palette colors (OSC 4), and whether it speaks
//! the Kitty graphics protocol (a query-only `a=q` command, which loads
//! nothing).
//!
//! The queries are followed by a Primary Device Attributes request
//! (`ESC [ c`), which practically every terminal answers. Terminals reply
//! in order, so once the DA1 reply arrives every color reply has arrived
//! too, or the terminal doesn't support them. Startup only waits the full
//! timeout on terminals that answer nothing at all.
//!
//! Must run in raw mode, before anything else reads the terminal
//! (crossterm's event reader would otherwise swallow the replies).

use std::time::{Duration, Instant};

/// Everything the startup query learned.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerminalReport {
    pub colors: TerminalColors,
    /// The terminal answered the Kitty graphics query with `OK`.
    pub kitty_graphics: bool,
}

/// What the terminal reported. Every field is optional: terminals answer
/// any subset.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerminalColors {
    pub foreground: Option<[u8; 3]>,
    pub background: Option<[u8; 3]>,
    /// ANSI colors 0-15, only when all 16 were reported.
    pub palette: Option<[[u8; 3]; 16]>,
}

/// How long to wait for the first replies (pi uses the same budget).
pub const QUERY_TIMEOUT: Duration = Duration::from_millis(100);
/// Once some replies have arrived, keep reading this long for the rest,
/// so late replies don't leak into the input stream as keystrokes.
const DRAIN_TIMEOUT: Duration = Duration::from_millis(1000);

fn query_bytes() -> Vec<u8> {
    let mut q = String::from("\x1b]10;?\x1b\\\x1b]11;?\x1b\\");
    for i in 0..16 {
        q.push_str(&format!("\x1b]4;{i};?\x1b\\"));
    }
    // Kitty graphics support: a 1x1 RGB query (`a=q` validates without
    // storing). Terminals without the protocol ignore the APC entirely.
    q.push_str(&format!("\x1b_Gi={KITTY_QUERY_ID},s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\"));
    // DA1 last: its reply marks the end of all replies.
    q.push_str("\x1b[c");
    q.into_bytes()
}

/// Image ID used only by the capability query.
const KITTY_QUERY_ID: u32 = 31;

/// Query the controlling terminal. Returns an empty report when there is
/// no terminal, or it doesn't answer.
#[cfg(unix)]
pub fn query() -> TerminalReport {
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;

    let Ok(mut tty) = std::fs::OpenOptions::new().read(true).write(true).open("/dev/tty") else {
        return TerminalReport::default();
    };
    if tty.write_all(&query_bytes()).and_then(|_| tty.flush()).is_err() {
        return TerminalReport::default();
    }

    let fd = tty.as_raw_fd();
    let start = Instant::now();
    let mut buf = Vec::with_capacity(512);
    let mut chunk = [0u8; 256];
    loop {
        if has_da1(&buf) {
            break;
        }
        let budget = if buf.is_empty() { QUERY_TIMEOUT } else { DRAIN_TIMEOUT };
        let Some(left) = budget.checked_sub(start.elapsed()) else { break };
        let mut pfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
        // SAFETY: `pfd` is a valid pollfd for the duration of the call.
        let ready = unsafe { libc::poll(&mut pfd, 1, left.as_millis().max(1) as libc::c_int) };
        if ready <= 0 {
            break; // timeout or error
        }
        match tty.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    let report = TerminalReport {
        colors: parse_replies(&buf),
        kitty_graphics: parse_kitty_reply(&buf),
    };
    tracing::info!(
        "terminal query: {} bytes in {:?}, fg={:?} bg={:?} palette={} kitty_graphics={}",
        buf.len(),
        start.elapsed(),
        report.colors.foreground,
        report.colors.background,
        report.colors.palette.is_some(),
        report.kitty_graphics
    );
    report
}

#[cfg(not(unix))]
pub fn query() -> TerminalReport {
    TerminalReport::default()
}

/// Did the terminal answer our graphics query with `OK`?
/// Reply: `ESC _G i=31 ; OK ESC \` (error replies carry a message instead).
fn parse_kitty_reply(buf: &[u8]) -> bool {
    let text = String::from_utf8_lossy(buf);
    let id = format!("i={KITTY_QUERY_ID}");
    text.split("\x1b_G").skip(1).any(|reply| {
        let reply = &reply[..reply.find('\x1b').unwrap_or(reply.len())];
        match reply.split_once(';') {
            Some((keys, msg)) => keys.split(',').any(|k| k == id) && msg == "OK",
            None => false,
        }
    })
}

/// A DA1 reply: `ESC [ ? … c`.
fn has_da1(buf: &[u8]) -> bool {
    buf.windows(3).enumerate().any(|(i, w)| {
        w == b"\x1b[?" && buf[i + 3..].iter().take_while(|b| **b != 0x1b).any(|b| *b == b'c')
    })
}

/// Parse every OSC 10/11/4 color reply in `buf`.
pub fn parse_replies(buf: &[u8]) -> TerminalColors {
    let text = String::from_utf8_lossy(buf);
    let mut colors = TerminalColors::default();
    let mut palette: [Option<[u8; 3]>; 16] = [None; 16];

    for reply in text.split("\x1b]").skip(1) {
        // Terminated by ST (ESC \) or BEL.
        let end = reply.find(['\x07', '\x1b']).unwrap_or(reply.len());
        let mut parts = reply[..end].split(';');
        match parts.next() {
            Some("10") => colors.foreground = parts.next().and_then(parse_color_spec),
            Some("11") => colors.background = parts.next().and_then(parse_color_spec),
            Some("4") => {
                let idx = parts.next().and_then(|i| i.parse::<usize>().ok());
                if let (Some(i @ 0..=15), Some(c)) = (idx, parts.next().and_then(parse_color_spec)) {
                    palette[i] = Some(c);
                }
            }
            _ => {}
        }
    }
    if palette.iter().all(Option::is_some) {
        colors.palette = Some(palette.map(Option::unwrap));
    }
    colors
}

/// X11 color spec: `rgb:R/G/B` (1-4 hex digits per channel), `rgba:…`,
/// or `#RRGGBB`.
fn parse_color_spec(spec: &str) -> Option<[u8; 3]> {
    let spec = spec.trim();
    if let Some(hex) = spec.strip_prefix('#') {
        if hex.len() == 6 {
            let n = u32::from_str_radix(hex, 16).ok()?;
            return Some([(n >> 16) as u8, (n >> 8) as u8, n as u8]);
        }
        return None;
    }
    let body = spec.strip_prefix("rgba:").or_else(|| spec.strip_prefix("rgb:"))?;
    let mut out = [0u8; 3];
    let mut channels = body.split('/');
    for slot in &mut out {
        let ch = channels.next()?;
        if ch.is_empty() || ch.len() > 4 {
            return None;
        }
        let v = u32::from_str_radix(ch, 16).ok()?;
        let max = (1u32 << (4 * ch.len())) - 1;
        *slot = ((v as f64 / max as f64) * 255.0).round() as u8;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_specs() {
        assert_eq!(parse_color_spec("rgb:ffff/8080/0000"), Some([255, 128, 0]));
        assert_eq!(parse_color_spec("rgb:ff/80/00"), Some([255, 128, 0]));
        assert_eq!(parse_color_spec("rgb:f/8/0"), Some([255, 136, 0]));
        assert_eq!(parse_color_spec("rgba:ffff/ffff/ffff/ffff"), Some([255, 255, 255]));
        assert_eq!(parse_color_spec("#1e1e2e"), Some([0x1e, 0x1e, 0x2e]));
        assert_eq!(parse_color_spec("rgb:zz/00/00"), None);
        assert_eq!(parse_color_spec("rgb:00/00"), None);
    }

    #[test]
    fn replies_st_and_bel() {
        let mut buf = String::from("\x1b]10;rgb:cdcd/d6d6/f4f4\x1b\\\x1b]11;rgb:1e1e/1e1e/2e2e\x07");
        for i in 0..16 {
            buf.push_str(&format!("\x1b]4;{i};rgb:{i:02x}/00/00\x1b\\"));
        }
        buf.push_str("\x1b[?62;22c");
        let c = parse_replies(buf.as_bytes());
        assert_eq!(c.foreground, Some([0xcd, 0xd6, 0xf4]));
        assert_eq!(c.background, Some([0x1e, 0x1e, 0x2e]));
        let p = c.palette.unwrap();
        assert_eq!(p[0], [0, 0, 0]);
        assert_eq!(p[15], [15, 0, 0]);
        assert!(has_da1(buf.as_bytes()));
    }

    #[test]
    fn partial_palette_is_dropped() {
        let c = parse_replies(b"\x1b]11;rgb:00/00/00\x1b\\\x1b]4;1;rgb:ff/00/00\x1b\\\x1b[?1;2c");
        assert_eq!(c.background, Some([0, 0, 0]));
        assert_eq!(c.foreground, None);
        assert_eq!(c.palette, None);
    }

    #[test]
    fn da1_detection() {
        assert!(!has_da1(b"\x1b]11;rgb:00/00/00\x1b\\"));
        assert!(!has_da1(b"\x1b[?62;22")); // incomplete
        assert!(has_da1(b"\x1b[?62;22c"));
        assert!(has_da1(b"\x1b[?6c"));
    }

    #[test]
    fn kitty_reply() {
        assert!(parse_kitty_reply(b"\x1b_Gi=31;OK\x1b\\\x1b[?62c"));
        assert!(!parse_kitty_reply(b"\x1b_Gi=31;EINVAL:bad\x1b\\"));
        assert!(!parse_kitty_reply(b"\x1b_Gi=7;OK\x1b\\"), "other image id");
        assert!(!parse_kitty_reply(b"\x1b[?62c"), "no reply at all");
    }

    #[test]
    fn query_ends_with_da1() {
        let q = query_bytes();
        assert!(q.ends_with(b"\x1b[c"));
        let graphics = q.windows(3).position(|w| w == b"\x1b_G").unwrap();
        assert!(graphics < q.len() - 3, "graphics query precedes DA1");
        assert_eq!(q.windows(4).filter(|w| w == b"\x1b]4;").count(), 16);
    }
}
