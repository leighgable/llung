//! Kitty graphics protocol, Unicode-placeholder flavour.
//!
//! Each image is transmitted once as a *virtual placement* (`U=1`) sized
//! in cells. It's then shown by writing placeholder characters
//! (U+10EEEE + row/column diacritics, image ID in the foreground color)
//! into ordinary buffer cells. To ratatui these are just styled text, so
//! diffing, scrolling, clearing and the Taffy layout work unchanged. No
//! cursor juggling, and no stale images when a cell is overwritten.
//!
//! Spec: https://sw.kovidgoyal.net/kitty/graphics-protocol/#unicode-placeholders
//! Verified against Ghostty's `src/terminal/kitty/graphics_unicode.zig`
//! (image ID from fg RGB, same diacritics table).

use base64::Engine;
use ratatui::buffer::Buffer;
use ratatui::style::Color;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// The placeholder codepoint.
const PLACEHOLDER: char = '\u{10EEEE}';

/// Base64 payload bytes per APC chunk (the protocol's maximum).
const CHUNK: usize = 4096;

/// Uploaded images and transmissions not yet written to the terminal.
///
/// Global because thumbnails are decoded from many places (presence,
/// history, attachments) that have no access to the terminal. The app
/// drains [`take_pending`] right before each frame is drawn.
struct Registry {
    next_id: u32,
    /// Source key → image ID, so re-decoding the same avatar is free.
    by_key: HashMap<u64, u32>,
    pending: Vec<u8>,
}

static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry {
        next_id: 1,
        by_key: HashMap::new(),
        pending: Vec::new(),
    })
});

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Upload an image (lazily encoded PNG) as a `cols` x `rows` virtual
/// placement and return its ID. `key` identifies the source and size:
/// the same key returns the existing ID without re-encoding or
/// re-sending. The terminal fits the image into the cell box, keeping
/// its aspect ratio.
pub fn upload(
    key: u64,
    cols: u16,
    rows: u16,
    encode_png: impl FnOnce() -> Option<Vec<u8>>,
) -> Option<u32> {
    let mut reg = registry();
    if let Some(&id) = reg.by_key.get(&key) {
        return Some(id);
    }
    // IDs travel in the 24-bit fg color.
    if reg.next_id > 0x00FF_FFFF {
        tracing::warn!("kitty image ID space exhausted");
        return None;
    }
    let png = encode_png()?;
    let id = reg.next_id;
    reg.next_id += 1;
    let seq = transmit_sequence(id, cols, rows, &png);
    reg.pending.extend_from_slice(&seq);
    reg.by_key.insert(key, id);
    Some(id)
}

/// Escape sequences waiting to be written (call before drawing a frame).
pub fn take_pending() -> Vec<u8> {
    std::mem::take(&mut registry().pending)
}

/// Escape sequences that free every image we uploaded (call on exit).
pub fn cleanup_sequence() -> Vec<u8> {
    let reg = registry();
    let mut out = Vec::new();
    for id in reg.by_key.values() {
        out.extend_from_slice(format!("\x1b_Ga=d,d=I,i={id},q=2\x1b\\").as_bytes());
    }
    out
}

/// `a=T` (transmit + display) with `U=1` (virtual placement), PNG data
/// sent directly, chunked. `q=2` suppresses replies, which would
/// otherwise arrive as input; later chunks inherit it.
fn transmit_sequence(id: u32, cols: u16, rows: u16, png: &[u8]) -> Vec<u8> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(png);
    let chunks: Vec<&[u8]> = b64.as_bytes().chunks(CHUNK).collect();
    let mut out = Vec::with_capacity(b64.len() + 64 * chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let more = u8::from(i + 1 < chunks.len());
        if i == 0 {
            out.extend_from_slice(
                format!("\x1b_Ga=T,U=1,f=100,t=d,q=2,i={id},c={cols},r={rows},m={more};").as_bytes(),
            );
        } else {
            out.extend_from_slice(format!("\x1b_Gm={more};").as_bytes());
        }
        out.extend_from_slice(chunk);
        out.extend_from_slice(b"\x1b\\");
    }
    out
}

/// Foreground color carrying an image ID.
pub fn id_color(id: u32) -> Color {
    Color::Rgb((id >> 16) as u8, (id >> 8) as u8, id as u8)
}

/// The cell content for row `row`, column `col` of an image.
pub fn placeholder(row: u16, col: u16) -> String {
    let mut s = String::with_capacity(12);
    s.push(PLACEHOLDER);
    s.push(diacritic(row));
    s.push(diacritic(col));
    s
}

fn diacritic(i: u16) -> char {
    let cp = DIACRITICS.get(i as usize).copied().unwrap_or(DIACRITICS[0]);
    char::from_u32(cp).unwrap_or('\u{0305}')
}

/// Draw image `id` as a `cols` x `rows` block of placeholder cells. The
/// cell background is left alone, so transparent image areas show the
/// themed surface below.
pub fn render(buf: &mut Buffer, x: u16, y: u16, id: u32, cols: u16, rows: u16) {
    let fg = id_color(id);
    for r in 0..rows {
        for c in 0..cols {
            if let Some(cell) = buf.cell_mut((x + c, y + r)) {
                cell.set_symbol(&placeholder(r, c));
                cell.set_fg(fg);
            }
        }
    }
}

/// Max image size in cells per axis that placeholders can address.
pub const MAX_CELLS: u16 = DIACRITICS.len() as u16;

/// Row/column diacritics: index = value. From kitty's
/// `gen/rowcolumn-diacritics.txt` (identical to Ghostty's copy).
#[rustfmt::skip]
const DIACRITICS: [u32; 297] = [
    0x0305, 0x030D, 0x030E, 0x0310, 0x0312, 0x033D, 0x033E, 0x033F, 0x0346, 0x034A,
    0x034B, 0x034C, 0x0350, 0x0351, 0x0352, 0x0357, 0x035B, 0x0363, 0x0364, 0x0365,
    0x0366, 0x0367, 0x0368, 0x0369, 0x036A, 0x036B, 0x036C, 0x036D, 0x036E, 0x036F,
    0x0483, 0x0484, 0x0485, 0x0486, 0x0487, 0x0592, 0x0593, 0x0594, 0x0595, 0x0597,
    0x0598, 0x0599, 0x059C, 0x059D, 0x059E, 0x059F, 0x05A0, 0x05A1, 0x05A8, 0x05A9,
    0x05AB, 0x05AC, 0x05AF, 0x05C4, 0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615,
    0x0616, 0x0617, 0x0657, 0x0658, 0x0659, 0x065A, 0x065B, 0x065D, 0x065E, 0x06D6,
    0x06D7, 0x06D8, 0x06D9, 0x06DA, 0x06DB, 0x06DC, 0x06DF, 0x06E0, 0x06E1, 0x06E2,
    0x06E4, 0x06E7, 0x06E8, 0x06EB, 0x06EC, 0x0730, 0x0732, 0x0733, 0x0735, 0x0736,
    0x073A, 0x073D, 0x073F, 0x0740, 0x0741, 0x0743, 0x0745, 0x0747, 0x0749, 0x074A,
    0x07EB, 0x07EC, 0x07ED, 0x07EE, 0x07EF, 0x07F0, 0x07F1, 0x07F3, 0x0816, 0x0817,
    0x0818, 0x0819, 0x081B, 0x081C, 0x081D, 0x081E, 0x081F, 0x0820, 0x0821, 0x0822,
    0x0823, 0x0825, 0x0826, 0x0827, 0x0829, 0x082A, 0x082B, 0x082C, 0x082D, 0x0951,
    0x0953, 0x0954, 0x0F82, 0x0F83, 0x0F86, 0x0F87, 0x135D, 0x135E, 0x135F, 0x17DD,
    0x193A, 0x1A17, 0x1A75, 0x1A76, 0x1A77, 0x1A78, 0x1A79, 0x1A7A, 0x1A7B, 0x1A7C,
    0x1B6B, 0x1B6D, 0x1B6E, 0x1B6F, 0x1B70, 0x1B71, 0x1B72, 0x1B73, 0x1CD0, 0x1CD1,
    0x1CD2, 0x1CDA, 0x1CDB, 0x1CE0, 0x1DC0, 0x1DC1, 0x1DC3, 0x1DC4, 0x1DC5, 0x1DC6,
    0x1DC7, 0x1DC8, 0x1DC9, 0x1DCB, 0x1DCC, 0x1DD1, 0x1DD2, 0x1DD3, 0x1DD4, 0x1DD5,
    0x1DD6, 0x1DD7, 0x1DD8, 0x1DD9, 0x1DDA, 0x1DDB, 0x1DDC, 0x1DDD, 0x1DDE, 0x1DDF,
    0x1DE0, 0x1DE1, 0x1DE2, 0x1DE3, 0x1DE4, 0x1DE5, 0x1DE6, 0x1DFE, 0x20D0, 0x20D1,
    0x20D4, 0x20D5, 0x20D6, 0x20D7, 0x20DB, 0x20DC, 0x20E1, 0x20E7, 0x20E9, 0x20F0,
    0x2CEF, 0x2CF0, 0x2CF1, 0x2DE0, 0x2DE1, 0x2DE2, 0x2DE3, 0x2DE4, 0x2DE5, 0x2DE6,
    0x2DE7, 0x2DE8, 0x2DE9, 0x2DEA, 0x2DEB, 0x2DEC, 0x2DED, 0x2DEE, 0x2DEF, 0x2DF0,
    0x2DF1, 0x2DF2, 0x2DF3, 0x2DF4, 0x2DF5, 0x2DF6, 0x2DF7, 0x2DF8, 0x2DF9, 0x2DFA,
    0x2DFB, 0x2DFC, 0x2DFD, 0x2DFE, 0x2DFF, 0xA66F, 0xA67C, 0xA67D, 0xA6F0, 0xA6F1,
    0xA8E0, 0xA8E1, 0xA8E2, 0xA8E3, 0xA8E4, 0xA8E5, 0xA8E6, 0xA8E7, 0xA8E8, 0xA8E9,
    0xA8EA, 0xA8EB, 0xA8EC, 0xA8ED, 0xA8EE, 0xA8EF, 0xA8F0, 0xA8F1, 0xAAB0, 0xAAB2,
    0xAAB3, 0xAAB7, 0xAAB8, 0xAABE, 0xAABF, 0xAAC1, 0xFE20, 0xFE21, 0xFE22, 0xFE23,
    0xFE24, 0xFE25, 0xFE26, 0x10A0F, 0x10A38, 0x1D185, 0x1D186, 0x1D187, 0x1D188, 0x1D189,
    0x1D1AA, 0x1D1AB, 0x1D1AC, 0x1D1AD, 0x1D242, 0x1D243, 0x1D244, 
];

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_width::UnicodeWidthStr;

    #[test]
    fn placeholder_is_one_cell_wide() {
        // ratatui lays cells out by display width; the combining
        // diacritics must not add any.
        assert_eq!(placeholder(0, 0).width(), 1);
        assert_eq!(placeholder(5, 290).width(), 1);
    }

    #[test]
    fn placeholder_encoding() {
        let p: Vec<char> = placeholder(1, 2).chars().collect();
        assert_eq!(p, vec!['\u{10EEEE}', '\u{030D}', '\u{030E}']);
        assert!(DIACRITICS.windows(2).all(|w| w[0] < w[1]), "sorted (Ghostty binary-searches)");
    }

    #[test]
    fn id_in_fg() {
        assert_eq!(id_color(0x0A0B0C), Color::Rgb(0x0A, 0x0B, 0x0C));
    }

    #[test]
    fn transmit_chunks() {
        let png = vec![7u8; 5000]; // > one chunk after base64
        let seq = String::from_utf8(transmit_sequence(9, 4, 2, &png)).unwrap();
        let apcs: Vec<&str> = seq.split("\x1b_G").skip(1).collect();
        assert_eq!(apcs.len(), 2);
        assert!(apcs[0].starts_with("a=T,U=1,f=100,t=d,q=2,i=9,c=4,r=2,m=1;"));
        assert!(apcs[1].starts_with("m=0;"));
        assert!(apcs.iter().all(|a| a.ends_with("\x1b\\")));
        let payload: String = apcs
            .iter()
            .map(|a| &a[a.find(';').unwrap() + 1..a.len() - 2])
            .collect();
        let decoded = base64::engine::general_purpose::STANDARD.decode(payload).unwrap();
        assert_eq!(decoded, png);
    }

    #[test]
    fn upload_dedups_by_key() {
        let key = 0xDEAD_BEEF_0000_0001;
        let a = upload(key, 4, 2, || Some(vec![1, 2, 3])).unwrap();
        let b = upload(key, 4, 2, || panic!("must not re-encode")).unwrap();
        assert_eq!(a, b);
        let pending = String::from_utf8(take_pending()).unwrap();
        assert!(pending.contains(&format!("i={a},")));
        assert!(cleanup_sequence().windows(4).any(|w| w == b"a=d,"));
    }

    #[test]
    fn render_writes_placeholders() {
        let mut buf = Buffer::empty(ratatui::layout::Rect::new(0, 0, 10, 4));
        render(&mut buf, 1, 1, 0x010203, 3, 2);
        let cell = &buf[(2, 2)];
        assert_eq!(cell.symbol(), placeholder(1, 1));
        assert_eq!(cell.fg, Color::Rgb(1, 2, 3));
        assert_eq!(buf[(0, 0)].symbol(), " ");
    }
}
