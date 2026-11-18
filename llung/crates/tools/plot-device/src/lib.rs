//! `plot_device` — a llung wasm tool.
//!
//! Plots tabular data (CSV) as terminal braille art: 2D or 3D scatter
//! plots, histograms, and CDFs, emitted as `braille-v1` render documents
//! (the same format `landscape` uses — viewers color by level with no
//! new code).
//!
//! Data in, art out, nothing else: the CSV arrives as the `csv` argument
//! (the chat host resolves `path` arguments to file contents before the
//! sandboxed call — wasm has no I/O).

wit_bindgen::generate!({
    path: "wit",
    world: "universal-tool",
});

mod braille;
mod csv;

/// KEEP IN SYNC with `PLOT_DEVICE_TOOL_SCHEMA` in crates/agent/src/tools.rs
/// (the host-side copy that goes into the LLM's system prompt).
/// (Only referenced by the component export glue, hence wasm-only.)
#[cfg(target_arch = "wasm32")]
const INPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "csv": {"type": "string", "description": "The CSV data itself. Omit when 'path' is given — the host reads the file and injects it."},
    "path": {"type": "string", "description": "Path to a CSV file on the host; the chat host reads it and injects the content as 'csv'. Use this instead of pasting data."},
    "kind": {"type": "string", "default": "scatter", "description": "Plot kind: scatter (default; 3D when 'z' is given), line, histogram, or cdf."},
    "x": {"type": "string", "description": "Domain column (scatter/line): header NAME or 0-based index. OMIT to use the row index (0,1,2,…) — the usual choice for a series over time."},
    "y": {"type": "string", "default": "1", "description": "Value column (scatter/line): name or index."},
    "z": {"type": "string", "description": "OPTIONAL column for z — 3D scatter with a pitch/yaw/distance camera. Omit for 2D."},
    "bins": {"type": "integer", "description": "Histogram bin count, 4..=200. Omit for one bin per data point (a bar per row)."},
    "pitch": {"type": "number", "default": 40, "description": "3D camera pitch (degrees, looking down)."},
    "yaw": {"type": "number", "default": 35, "description": "3D camera yaw (degrees)."},
    "distance": {"type": "number", "default": 3.5, "description": "3D camera distance (must be > 0)."},
    "width": {"type": "integer", "default": 120, "description": "Output width in characters, 8..=400 (width*height is capped at 10000 cells)"},
    "height": {"type": "integer", "default": 30, "description": "Output height in characters, 4..=200; defaults to a letterbox 30 (width*height is capped at 10000 cells)"}
  }
}"#;

pub const MAX_CELLS: usize = 10_000;
/// The host will not inline files larger than this into arguments.
pub const MAX_CSV_BYTES: usize = 256 * 1024;

#[cfg(target_arch = "wasm32")]
struct PlotDevice;

#[cfg(target_arch = "wasm32")]
impl Guest for PlotDevice {
    fn get_metadata() -> ToolMetadata {
        ToolMetadata {
            name: "plot_device".into(),
            description: "Plot tabular data (CSV) as terminal braille: scatter \
                          plots (2D or 3D), histograms, and CDFs. Use when asked \
                          to plot, chart, or visualize data, columns, or a CSV file."
                .into(),
            input_schema: INPUT_SCHEMA.into(),
        }
    }

    fn execute(arguments_json: String) -> Result<String, String> {
        run(&arguments_json)
    }
}

#[cfg(target_arch = "wasm32")]
export!(PlotDevice);

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Scatter,
    Line,
    Histogram,
    Cdf,
}

struct Args {
    csv: String,
    kind: Kind,
    /// Domain column; None = row index.
    x: Option<String>,
    y: String,
    z: Option<String>,
    /// Histogram bins; None = one per data point.
    bins: Option<usize>,
    camera: Camera,
    width: usize,
    height: usize,
}

/// The x values: the selected column, or the row index when none is
/// given (the usual domain for a series).
fn domain(table: &csv::Table, args: &Args, n: usize) -> Result<(Vec<f64>, String), String> {
    match &args.x {
        Some(sel) => Ok((table.column(sel)?, sel.clone())),
        None => Ok(((0..n).map(|i| i as f64).collect(), "index".into())),
    }
}

fn parse_args(arguments_json: &str) -> Result<Args, String> {
    let v: serde_json::Value = serde_json::from_str(arguments_json)
        .map_err(|e| format!("arguments must be a JSON object: {e}"))?;
    if !v.is_object() {
        return Err("arguments must be a JSON object".into());
    }
    let str_arg = |key: &str| v.get(key).and_then(|s| s.as_str()).map(str::to_string);

    let csv = str_arg("csv")
        .or_else(|| str_arg("path"))
        .ok_or("missing required 'csv' (or a 'path' the host can read)")?;
    if csv.len() > MAX_CSV_BYTES {
        return Err(format!(
            "csv data is {} bytes; the cap is {} — plot a smaller excerpt",
            csv.len(),
            MAX_CSV_BYTES
        ));
    }

    let kind = match str_arg("kind").unwrap_or_else(|| "scatter".into()).to_ascii_lowercase().as_str() {
        "scatter" | "points" | "pointcloud" | "point_cloud" => Kind::Scatter,
        "line" | "lines" | "timeseries" | "time_series" => Kind::Line,
        "histogram" | "hist" | "barchart" | "bar" => Kind::Histogram,
        "cdf" => Kind::Cdf,
        other => return Err(format!("unknown kind '{other}' (try: scatter, line, histogram, cdf)")),
    };

    // Letterbox default: a wide strip, not a full-screen panel.
    let (mut width, mut height) = (
        v.get("width").and_then(|n| n.as_i64()).unwrap_or(120).clamp(8, 400) as usize,
        v.get("height").and_then(|n| n.as_i64()).unwrap_or(30).clamp(4, 200) as usize,
    );
    if width * height > MAX_CELLS {
        let scale = (MAX_CELLS as f64 / (width * height) as f64).sqrt();
        width = ((width as f64 * scale).floor() as usize).max(8);
        height = ((height as f64 * scale).floor() as usize).max(4);
    }

    let num = |key: &str, default: f64, min: f64, max: f64| {
        v.get(key)
            .and_then(|n| n.as_f64())
            .unwrap_or(default)
            .clamp(min, max) as f32
    };

    Ok(Args {
        csv,
        kind,
        x: str_arg("x"),
        y: str_arg("y").unwrap_or_else(|| "1".into()),
        z: str_arg("z"),
        bins: v
            .get("bins")
            .and_then(|n| n.as_i64())
            .map(|b| b.clamp(4, 200) as usize),
        camera: Camera {
            pitch_rad: num("pitch", 40.0, -89.0, 89.0).to_radians(),
            yaw_rad: num("yaw", 35.0, -360.0, 360.0).to_radians(),
            distance: num("distance", 3.5, 0.1, 100.0),
        },
        width,
        height,
    })
}

/// The same yaw/pitch/perspective pipeline as the landscape tool.
struct Camera {
    pitch_rad: f32,
    yaw_rad: f32,
    distance: f32,
}

impl Camera {
    fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let (cp, sp) = (self.pitch_rad.cos(), self.pitch_rad.sin());
        let (cy, sy) = (self.yaw_rad.cos(), self.yaw_rad.sin());
        let x_rot = x * cy - y * sy;
        let y_temp = x * sy + y * cy;
        let y_rot = y_temp * cp - z * sp;
        let z_rot = y_temp * sp + z * cp;
        let z_final = z_rot + self.distance;
        (x_rot / z_final, y_rot / z_final)
    }
}

/// Render and return a `braille-v1` document (see `landscape` for the
/// viewer contract). Segments carry the COLOR variable in their z: the
/// data z for 3D scatter, y for 2D scatter, bar height for histograms,
/// cumulative fraction for CDFs.
pub fn run(arguments_json: &str) -> Result<String, String> {
    let args = parse_args(arguments_json)?;
    let table = csv::parse(&args.csv)?;

    let (segments, title, color): (Vec<[braille::Point; 2]>, String, String) = match args.kind {
        Kind::Scatter | Kind::Line => {
            let ys = table.column(&args.y)?;
            if ys.is_empty() {
                return Err(format!("no numeric values in column '{}'", args.y));
            }
            let (xs, xname) = domain(&table, &args, ys.len())?;
            let n = xs.len().min(ys.len());
            if n == 0 {
                return Err("no rows with numeric x and y".into());
            }
            let kind_name = if args.kind == Kind::Scatter { "scatter" } else { "line" };
            match &args.z {
                Some(sel) if args.kind == Kind::Scatter => {
                    let zs = table.column(sel)?;
                    let n = n.min(zs.len());
                    let segs: Vec<[braille::Point; 2]> = (0..n)
                        .map(|i| {
                            let (px, py) = args
                                .camera
                                .project(xs[i] as f32, ys[i] as f32, zs[i] as f32);
                            let p = (px, py, zs[i] as f32);
                            [p, p]
                        })
                        .collect();
                    (
                        segs,
                        format!("plot_device:scatter3d:{xname}×{}×{sel}", args.y),
                        sel.clone(),
                    )
                }
                _ if args.kind == Kind::Line => {
                    let pts: Vec<braille::Point> = (0..n)
                        .map(|i| (xs[i] as f32, ys[i] as f32, ys[i] as f32))
                        .collect();
                    let segs: Vec<[braille::Point; 2]> =
                        pts.iter().zip(pts.iter().skip(1)).map(|(&a, &b)| [a, b]).collect();
                    (
                        segs,
                        format!("plot_device:line:{xname}×{}", args.y),
                        args.y.clone(),
                    )
                }
                _ => {
                    let segs: Vec<[braille::Point; 2]> = (0..n)
                        .map(|i| {
                            let p = (xs[i] as f32, ys[i] as f32, ys[i] as f32);
                            [p, p]
                        })
                        .collect();
                    (
                        segs,
                        format!("plot_device:{kind_name}:{xname}×{}", args.y),
                        args.y.clone(),
                    )
                }
            }
        }
        Kind::Histogram | Kind::Cdf => {
            // The value column: x by convention, y as an accepted alias,
            // else the first column. (The domain index doesn't apply to
            // value distributions.)
            let value_sel = args
                .x
                .clone()
                .or_else(|| Some(args.y.clone()))
                .unwrap_or_else(|| "0".into());
            let xs = table.column(&value_sel)?;
            if xs.is_empty() {
                return Err(format!("no numeric values in column '{value_sel}'"));
            }
            if args.kind == Kind::Cdf {
                let mut sorted = xs.clone();
                sorted.sort_by(f64::total_cmp);
                let n = sorted.len();
                let segs: Vec<[braille::Point; 2]> = sorted
                    .iter()
                    .enumerate()
                    .zip(sorted.iter().enumerate().skip(1))
                    .map(|((i, &a), (_, &b))| {
                        let y0 = i as f32 / n as f32;
                        let y1 = (i + 1) as f32 / n as f32;
                        [(a as f32, y0, y0), (b as f32, y1, y1)]
                    })
                    .collect();
                (
                    segs,
                    format!("plot_device:cdf:{value_sel}"),
                    "cdf".to_string(),
                )
            } else {
                let bins = args.bins.unwrap_or_else(|| xs.len().clamp(4, 200));
                let (lo, hi) = xs.iter().fold(
                    (f64::INFINITY, f64::NEG_INFINITY),
                    |(l, h), &v| (l.min(v), h.max(v)),
                );
                let span = (hi - lo).max(1e-12);
                let mut counts = vec![0usize; bins];
                for &v in &xs {
                    let b = (((v - lo) / span) * bins as f64)
                        .floor()
                        .min(bins as f64 - 1.0)
                        .max(0.0) as usize;
                    counts[b] += 1;
                }
                let max = counts.iter().copied().max().unwrap_or(1).max(1);
                let segs: Vec<[braille::Point; 2]> = counts
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| {
                        let x = i as f32 + 0.5;
                        let h = c as f32 / max as f32;
                        [(x, 0.0, h), (x, h, h)]
                    })
                    .collect();
                (
                    segs,
                    format!("plot_device:histogram:{value_sel} ({bins} bins)"),
                    format!("count, max {max}"),
                )
            }
        }
    };

    let raster = braille::rasterize(&segments, args.width, args.height);
    let doc = serde_json::json!({
        "render": "braille-v1",
        "title": title,
        "width": raster.width,
        "rows": raster.rows,
        "levels": raster.levels,
        "z_range": [0.0, 1.0],
        "color_hint": color,
    });
    Ok(doc.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(args: serde_json::Value) -> serde_json::Value {
        let out = run(&serde_json::json!(args).to_string()).expect("args should work");
        serde_json::from_str(&out).expect("valid JSON")
    }

    fn drawn(d: &serde_json::Value) -> usize {
        d["rows"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_str().unwrap().chars())
            .filter(|c| *c != '\u{2800}')
            .count()
    }

    const CSV: &str = "age,height,weight\n30,170,70\n40,180,80\n25,165,55\n35,175,75\n50,172,90\n22,168,50\n";

    #[test]
    fn scatter_2d_by_column_names() {
        let d = doc(serde_json::json!({
            "csv": CSV, "kind": "scatter", "x": "age", "y": "height",
            "width": 40, "height": 16
        }));
        assert_eq!(d["render"], "braille-v1");
        assert!(d["title"].as_str().unwrap().contains("scatter"));
        assert!(drawn(&d) >= 5, "one dot per row: {}", drawn(&d));
    }

    #[test]
    fn scatter_3d_uses_camera_and_z_levels() {
        let d = doc(serde_json::json!({
            "csv": CSV, "kind": "scatter", "x": "age", "y": "height", "z": "weight",
            "width": 40, "height": 16
        }));
        assert!(d["title"].as_str().unwrap().contains("scatter3d"));
        assert!(drawn(&d) >= 5, "{}", drawn(&d));
        let levels: String = d["levels"].as_array().unwrap().iter()
            .flat_map(|l| l.as_str().unwrap().chars()).collect();
        assert!(levels.chars().any(|c| c.is_ascii_digit()), "colored by z: {levels}");
    }

    #[test]
    fn histogram_bins_draw_bars() {
        let d = doc(serde_json::json!({
            "csv": CSV, "kind": "histogram", "x": "age", "bins": 8,
            "width": 40, "height": 16
        }));
        assert!(d["title"].as_str().unwrap().contains("histogram"));
        // Bars are vertical segments: expect more dots than points.
        assert!(drawn(&d) >= 8, "{}", drawn(&d));
    }

    #[test]
    fn line_plot_connects_points_over_the_index() {
        let d = doc(serde_json::json!({
            "csv": CSV, "kind": "line", "y": "weight",
            "width": 40, "height": 12
        }));
        // No x given → the domain is the row index.
        assert!(
            d["title"].as_str().unwrap().contains("line:index×weight"),
            "{}",
            d["title"]
        );
        // A connected line draws more dots than the 6 raw points.
        assert!(drawn(&d) >= 12, "{}", drawn(&d));
    }

    #[test]
    fn histogram_default_bins_is_one_per_row() {
        // 6 data rows, no bins → 6 bins (a bar per row).
        let d = doc(serde_json::json!({
            "csv": CSV, "kind": "histogram", "x": "weight",
            "width": 40, "height": 12
        }));
        assert!(
            d["title"].as_str().unwrap().contains("(6 bins)"),
            "{}",
            d["title"]
        );
        assert!(drawn(&d) >= 6, "{}", drawn(&d));
    }

    #[test]
    fn cdf_line_is_monotonic_enough() {
        let d = doc(serde_json::json!({
            "csv": CSV, "kind": "cdf", "x": "age", "width": 40, "height": 16
        }));
        assert!(d["title"].as_str().unwrap().contains("cdf"));
        assert!(drawn(&d) >= 4, "{}", drawn(&d));
    }

    #[test]
    fn headerless_csv_by_index() {
        let d = doc(serde_json::json!({
            "csv": "1,10\n2,20\n3,30\n", "kind": "scatter", "width": 24, "height": 10
        }));
        assert!(drawn(&d) >= 3);
    }

    #[test]
    fn errors_are_clean() {
        assert!(run("{}").unwrap_err().contains("csv"));
        let bad_kind = serde_json::json!({"csv": CSV, "kind": "pie"}).to_string();
        assert!(run(&bad_kind).unwrap_err().contains("unknown kind"));
        let bad_col = serde_json::json!({"csv": CSV, "x": "nope", "y": "height"}).to_string();
        assert!(run(&bad_col).unwrap_err().contains("unknown column"));
        // A text column as the VALUE gives a clean, specific error.
        let flights = "year,month,passengers\n1949,January,112\n";
        let bad_val = serde_json::json!({"csv": flights, "kind": "line", "y": "month"}).to_string();
        assert!(
            run(&bad_val)
                .unwrap_err()
                .contains("no numeric values in column 'month'")
        );
    }
}
