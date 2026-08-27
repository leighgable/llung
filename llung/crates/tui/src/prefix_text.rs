use unicode_width::UnicodeWidthChar;

/// Text with O(1) width queries and O(log n) line fitting.
pub struct PrefixText {
    text: String,
    /// Byte index of each char boundary
    char_indices: Vec<usize>,
    /// Cumulative display width up to each char index
    cumulative: Vec<u16>,
}

impl PrefixText {
    pub fn new(text: String) -> Self {
        let mut char_indices = Vec::with_capacity(text.len());
        let mut cumulative = vec![0u16];

        for (idx, ch) in text.char_indices() {
            char_indices.push(idx);
            let w = ch.width().unwrap_or(0) as u16;
            cumulative.push(cumulative.last().unwrap().saturating_add(w));
        }

        Self {
            text,
            char_indices,
            cumulative,
        }
    }

    /// Display width of chars [start .. end].
    fn slice_width(&self, start: usize, end: usize) -> u16 {
        self.cumulative[end] - self.cumulative[start]
    }

    /// Largest char index such that width from `start` fits in `max_width`.
    fn fit_chars(&self, start: usize, max_width: u16) -> usize {
        let base = self.cumulative[start];
        let target = base.saturating_add(max_width);
        let mut lo = start;
        let mut hi = self.char_indices.len();

        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.cumulative[mid] <= target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.max(start + 1)
    }

    /// Iterate over visual lines wrapped to `max_width` columns.
    pub fn wrap_lines(&self, max_width: u16) -> impl Iterator<Item = &str> + '_ {
        let mut start_char = 0usize;
        let char_count = self.char_indices.len();

        std::iter::from_fn(move || {
            if start_char >= char_count {
                return None;
            }
            let end_char = self.fit_chars(start_char, max_width);
            let byte_start = self.char_indices[start_char];
            let byte_end = if end_char < char_count {
                self.char_indices[end_char]
            } else {
                self.text.len()
            };
            let line = &self.text[byte_start..byte_end];
            start_char = end_char;
            Some(line)
        })
    }
}
