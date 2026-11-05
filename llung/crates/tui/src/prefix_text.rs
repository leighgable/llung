use unicode_width::UnicodeWidthChar;

/// Text with O(1) width queries and O(log n) line fitting.
///
/// `'\n'` is a hard line break; paragraphs between breaks are wrapped
/// independently.
pub struct PrefixText {
    text: String,
    /// Byte index of each char boundary
    char_indices: Vec<usize>,
    /// Cumulative display width up to each char index
    cumulative: Vec<u16>,
    /// Char indices of `'\n'`, ascending.
    newlines: Vec<usize>,
}

impl PrefixText {
    pub fn new(text: String) -> Self {
        let mut char_indices = Vec::with_capacity(text.len());
        let mut cumulative = vec![0u16];
        let mut newlines = Vec::new();

        for (i, (idx, ch)) in text.char_indices().enumerate() {
            char_indices.push(idx);
            if ch == '\n' {
                newlines.push(i);
            }
            let w = ch.width().unwrap_or(0) as u16;
            cumulative.push(cumulative.last().unwrap().saturating_add(w));
        }

        Self {
            text,
            char_indices,
            cumulative,
            newlines,
        }
    }

    /// Display width of chars [start .. end].
    #[allow(dead_code)]
    fn slice_width(&self, start: usize, end: usize) -> u16 {
        self.cumulative[end] - self.cumulative[start]
    }

    /// Largest char index `k` in `start..=limit` such that chars
    /// `[start, k)` fit in `max_width`. Always > `start` (a single char
    /// wider than the line is emitted alone rather than looping forever).
    /// Zero-width chars (e.g. combining marks) stay with their base char.
    fn fit_chars(&self, start: usize, limit: usize, max_width: u16) -> usize {
        let target = self.cumulative[start].saturating_add(max_width);
        // cumulative[start..=limit] is non-decreasing; count entries <= target.
        let fitting = self.cumulative[start..=limit].partition_point(|&c| c <= target);
        (start + fitting - 1).max(start + 1).min(limit.max(start + 1))
    }

    /// Char index of the first `'\n'` at or after `start` (or the end).
    fn paragraph_end(&self, start: usize) -> usize {
        let i = self.newlines.partition_point(|&n| n < start);
        self.newlines.get(i).copied().unwrap_or(self.char_indices.len())
    }

    fn slice(&self, start: usize, end: usize) -> &str {
        let byte_start = self.char_indices.get(start).copied().unwrap_or(self.text.len());
        let byte_end = self.char_indices.get(end).copied().unwrap_or(self.text.len());
        &self.text[byte_start..byte_end]
    }

    /// Iterate over visual lines wrapped to `max_width` columns. Lines
    /// never contain `'\n'`; empty paragraphs yield empty lines, and a
    /// single trailing newline doesn't add one.
    pub fn wrap_lines(&self, max_width: u16) -> impl Iterator<Item = &str> + '_ {
        let mut start = 0usize;
        let char_count = self.char_indices.len();

        std::iter::from_fn(move || {
            if start >= char_count {
                return None;
            }
            let para_end = self.paragraph_end(start);
            if start == para_end {
                // Empty paragraph: consume the '\n'.
                start += 1;
                return Some("");
            }
            let end = self.fit_chars(start, para_end, max_width);
            let line = self.slice(start, end);
            // Step over the paragraph's '\n' once its last line is emitted.
            start = if end == para_end && para_end < char_count { end + 1 } else { end };
            Some(line)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str, width: u16) -> Vec<String> {
        PrefixText::new(text.into()).wrap_lines(width).map(String::from).collect()
    }

    #[test]
    fn wraps_at_width_exactly() {
        // Regression: fit_chars used to return one char too many.
        assert_eq!(lines("abcdefgh", 3), ["abc", "def", "gh"]);
        assert_eq!(lines("abc", 3), ["abc"]);
        assert_eq!(lines("", 3), Vec::<String>::new());
    }

    #[test]
    fn newlines_are_hard_breaks() {
        assert_eq!(lines("ab\ncd", 10), ["ab", "cd"]);
        assert_eq!(lines("abcd\nef", 2), ["ab", "cd", "ef"]);
        assert_eq!(lines("a\n\nb", 10), ["a", "", "b"]);
        assert_eq!(lines("a\n", 10), ["a"]);
        assert_eq!(lines("\na", 10), ["", "a"]);
    }

    #[test]
    fn braille_rows_stay_rows() {
        let chart = "⠁⠂⠃⠇\n⠏⠟⠿⣿\n";
        assert_eq!(lines(chart, 80), ["⠁⠂⠃⠇", "⠏⠟⠿⣿"]);
    }

    #[test]
    fn wide_and_combining_chars() {
        // A wide char wider than the line is emitted alone.
        assert_eq!(lines("漢字", 1), ["漢", "字"]);
        assert_eq!(lines("漢字", 3), ["漢", "字"]);
        // Combining mark stays with its base.
        assert_eq!(lines("e\u{301}x", 1), ["e\u{301}", "x"]);
    }
}
