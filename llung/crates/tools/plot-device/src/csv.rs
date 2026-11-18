//! Minimal CSV handling for `plot_device`.
//!
//! Deliberately small: comma-separated, no quoted-field support (yet),
//! `#` comment lines skipped. If the first row contains any non-numeric
//! cell it becomes the header so columns can be selected by NAME as
//! well as by 0-based index — the model reliably knows both.
//!
//! Only the *selected* columns need to be numeric: a text column
//! (month names, labels) in an unused position keeps its rows usable.

#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    /// Column names when the CSV had a header row.
    pub header: Option<Vec<String>>,
    /// Cells per column, in row order; non-numeric cells are None.
    pub columns: Vec<Vec<Option<f64>>>,
    /// Well-formed data rows (ragged rows excluded).
    pub rows: usize,
    /// Rows dropped because the cell count didn't match.
    pub skipped_rows: usize,
}

impl Table {
    /// The numeric values of a column, in row order — non-numeric cells
    /// are dropped, never disqualifying their whole row.
    pub fn column(&self, selector: &str) -> Result<Vec<f64>, String> {
        let col: &[Option<f64>] = if let Some(i) = parse_index(selector) {
            self.columns.get(i).map(Vec::as_slice).ok_or_else(|| {
                format!(
                    "column index {i} out of range ({} columns)",
                    self.columns.len()
                )
            })?
        } else {
            let header = self.header.as_ref().ok_or_else(|| {
                format!(
                    "no header row — select columns by index (0..{})",
                    self.columns.len()
                )
            })?;
            let i = header
                .iter()
                .position(|h| h.eq_ignore_ascii_case(selector.trim()))
                .ok_or_else(|| {
                    format!(
                        "unknown column '{selector}' (have: {})",
                        header.join(", ")
                    )
                })?;
            &self.columns[i]
        };
        Ok(col.iter().flatten().copied().collect())
    }
}

fn parse_index(selector: &str) -> Option<usize> {
    let t = selector.trim();
    if t.chars().all(|c| c.is_ascii_digit()) && !t.is_empty() {
        return t.parse().ok();
    }
    None
}

pub fn parse(csv: &str) -> Result<Table, String> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in csv.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        rows.push(t.split(',').map(|c| c.trim().to_string()).collect());
    }
    let Some(first) = rows.first() else {
        return Err("the CSV is empty".into());
    };
    let width = first.len();
    if width == 0 {
        return Err("the CSV has no columns".into());
    }

    // Header when any first-row cell is non-numeric.
    let has_header = first.iter().any(|c| c.parse::<f64>().is_err());
    let header = if has_header {
        Some(first.clone())
    } else {
        None
    };

    let mut columns: Vec<Vec<Option<f64>>> = vec![Vec::new(); width];
    let mut skipped = 0usize;
    let mut data_rows = 0usize;
    for row in rows.iter().skip(if has_header { 1 } else { 0 }) {
        if row.len() != width {
            skipped += 1;
            continue;
        }
        data_rows += 1;
        for (i, cell) in row.iter().enumerate() {
            columns[i].push(match cell.parse::<f64>() {
                Ok(v) if v.is_finite() => Some(v),
                _ => None,
            });
        }
    }
    if data_rows == 0 {
        return Err("no data rows in the CSV".into());
    }
    Ok(Table {
        header,
        columns,
        rows: data_rows,
        skipped_rows: skipped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_detection_and_name_selection() {
        let t = parse("age,height\n30,170\n40,180\nbad,row\n25,165\n").unwrap();
        assert_eq!(t.header.as_ref().unwrap(), &vec!["age", "height"]);
        assert_eq!(t.column("height").unwrap(), &[170.0, 180.0, 165.0]);
        // The "bad,row" cells are dropped per-cell: age lost "bad",
        // but 30/40/25 survive.
        assert_eq!(t.column("age").unwrap(), &[30.0, 40.0, 25.0]);
        assert!(t.column("weight").unwrap_err().contains("unknown column"));
    }

    #[test]
    fn text_columns_do_not_disqualify_rows() {
        // The AirPassengers shape: a text column (month) alongside
        // numeric ones — every row must stay usable for year/passengers.
        let t = parse("year,month,passengers\n1949,January,112\n1949,February,118\n1950,March,132\n").unwrap();
        assert_eq!(t.rows, 3);
        assert_eq!(t.column("year").unwrap(), &[1949.0, 1949.0, 1950.0]);
        assert_eq!(t.column("passengers").unwrap(), &[112.0, 118.0, 132.0]);
        // Selecting the text column yields no numbers — a clean error
        // surfaced by the caller, not a silent empty plot.
        assert!(t.column("month").unwrap().is_empty());
    }

    #[test]
    fn headerless_numeric_csv_selects_by_index() {
        let t = parse("1,2\n3,4\n5,6\n").unwrap();
        assert!(t.header.is_none());
        assert_eq!(t.column("1").unwrap(), &[2.0, 4.0, 6.0]);
        assert!(t.column("x").is_err());
    }

    #[test]
    fn empty_and_jagged_csv_errors() {
        assert!(parse("").is_err());
        assert!(parse("# only comments\n").is_err());
        let t = parse("a,b\n1,2\n3\n").unwrap();
        assert_eq!(t.skipped_rows, 1);
        assert_eq!(t.rows, 1);
    }
}
