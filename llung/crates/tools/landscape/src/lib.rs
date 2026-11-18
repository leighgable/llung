//! `landscape` — a llung wasm tool.
//!
//! Renders a 3D wireframe surface of a math expression `f(x, y)` as
//! terminal braille art, returned as a `braille-v1` render document (see
//! [`run`]) so viewers can fit and color it. The model controls the math and the camera via
//! JSON arguments; the expression evaluator (`expr`) is the safe
//! alternative to a code sandbox: no I/O, no side effects, bounded cost.

wit_bindgen::generate!({
    path: "wit",
    world: "universal-tool",
});

mod braille;
mod expr;

// `Guest` and `ToolMetadata` come from the `generate!` macro above — it
// emits them at the invocation site (crate root).
use expr::Expr;

/// KEEP IN SYNC with `LANDSCAPE_TOOL_SCHEMA` in crates/agent/src/tools.rs
/// (the host-side copy that goes into the LLM's system prompt).
/// (Only referenced by the component export glue, hence wasm-only.)
#[cfg(target_arch = "wasm32")]
const INPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "expr": {
      "type": "string",
      "default": "cos(sqrt(x^2+y^2)*3)*0.4",
      "description": "f(x,y) expression. Variables: x, y. Operators: + - * / % ^. Functions: sin cos tan asin acos atan atan2 sqrt abs exp ln log2 log10 floor ceil round min max. Constants: pi tau e."
    },
    "pitch": {"type": "number", "default": 40, "description": "OPTIONAL camera pitch in degrees (looking down). Omit unless the user asks to adjust the view; values near 0 look edge-on and unreadable."},
    "yaw": {"type": "number", "default": 35, "description": "OPTIONAL camera yaw in degrees. Omit unless asked."},
    "distance": {"type": "number", "default": 3.5, "description": "OPTIONAL camera distance (must be > 0). Omit unless asked; values near 1 give extreme distortion."},
    "grid": {"type": "integer", "default": 15, "description": "OPTIONAL mesh resolution: 4..=200 for heightfield mode; 4..=64 for implicit (z) mode, where it applies PER AXIS — keep it modest (25-40). Omit unless asked."},
    "width": {"type": "integer", "default": 120, "description": "OPTIONAL output width in characters, 8..=400 (width*height is capped at 10000 cells). Omit unless asked."},
    "height": {"type": "integer", "default": 60, "description": "OPTIONAL output height in characters, 4..=200 (width*height is capped at 10000 cells). Omit unless asked."}
  }
}"#;

const DEFAULT_EXPR: &str = "cos(sqrt(x^2+y^2)*3)*0.4";

/// Output cap. Results travel as gossipsub chat messages (64 KiB max);
/// a cell costs ~4 bytes (3-byte braille char + 1 level byte), so 10k
/// cells is ~40 KiB, leaving room for the message envelope.
pub const MAX_CELLS: usize = 10_000;

#[cfg(target_arch = "wasm32")]
struct Landscape;

#[cfg(target_arch = "wasm32")]
impl Guest for Landscape {
    fn get_metadata() -> ToolMetadata {
        ToolMetadata {
            name: "landscape".into(),
            description: "Render a 3D wireframe surface of a math expression f(x,y) as terminal \
                          braille art. Use when asked to plot, chart, graph, or visualize a \
                          function or landscape."
                .into(),
            input_schema: INPUT_SCHEMA.into(),
        }
    }

    fn execute(arguments_json: String) -> Result<String, String> {
        run(&arguments_json)
    }
}

#[cfg(target_arch = "wasm32")]
export!(Landscape);

/// Heightfield mode: sample `z = f(x, y)` over [-2, 2]^2 and emit
/// wireframe edges between grid neighbours.
fn heightfield_segments(expr: &Expr, camera: &Camera, n: usize) -> Vec<[braille::Point; 2]> {
    let n = n.clamp(4, 200);
    let mut mesh = vec![vec![(0.0f32, 0.0f32, 0.0f32); n]; n];
    for i in 0..n {
        for j in 0..n {
            let x = (i as f64 / (n - 1) as f64) * 4.0 - 2.0;
            let y = (j as f64 / (n - 1) as f64) * 4.0 - 2.0;
            let z = expr.eval(x, y, 0.0);
            // Keep poles/asymptotes from swallowing the view.
            let z = if z.is_finite() { z.clamp(-10.0, 10.0) } else { 0.0 };
            let (px, py) = camera.project(x as f32, y as f32, z as f32);
            mesh[i][j] = (px, py, z as f32);
        }
    }
    let mut segments = Vec::with_capacity(n * (n - 1) * 2);
    for i in 0..n {
        for j in 0..n {
            if i + 1 < n {
                segments.push([mesh[i][j], mesh[i + 1][j]]);
            }
            if j + 1 < n {
                segments.push([mesh[i][j], mesh[i][j + 1]]);
            }
        }
    }
    segments
}

/// Implicit-surface mode: sample `f(x, y, z)` on an `n`³ grid over
/// [-2, 2]³ and emit a point (as a zero-length segment) wherever the
/// sign changes along a grid edge — linearly interpolated. The points
/// keep their world `z` so viewers can color by height (a torus comes
/// out with a nice vertical gradient).
fn iso_points(expr: &Expr, camera: &Camera, n: usize) -> Vec<[braille::Point; 2]> {
    let n = n.clamp(4, 64);
    let coord = |i: usize| (i as f64 / (n - 1) as f64) * 4.0 - 2.0;
    let idx = |i: usize, j: usize, k: usize| (k * n + j) * n + i;

    let mut v = vec![f64::NAN; n * n * n];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                v[idx(i, j, k)] = expr.eval(coord(i), coord(j), coord(k));
            }
        }
    }

    /// Fraction along an edge where the linearized field crosses zero.
    fn crossing(a: f64, b: f64) -> Option<f64> {
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        if a == 0.0 {
            return Some(0.0);
        }
        if a.signum() != b.signum() {
            Some(a / (a - b)).filter(|t| (0.0..=1.0).contains(t))
        } else {
            None
        }
    }

    let mut pts: Vec<(f64, f64, f64)> = Vec::new();
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let f = v[idx(i, j, k)];
                if i + 1 < n {
                    if let Some(t) = crossing(f, v[idx(i + 1, j, k)]) {
                        pts.push((coord(i) + t * (coord(i + 1) - coord(i)), coord(j), coord(k)));
                    }
                }
                if j + 1 < n {
                    if let Some(t) = crossing(f, v[idx(i, j + 1, k)]) {
                        pts.push((coord(i), coord(j) + t * (coord(j + 1) - coord(j)), coord(k)));
                    }
                }
                if k + 1 < n {
                    if let Some(t) = crossing(f, v[idx(i, j, k + 1)]) {
                        pts.push((coord(i), coord(j), coord(k) + t * (coord(k + 1) - coord(k))));
                    }
                }
            }
        }
    }

    pts.into_iter()
        .map(|(x, y, z)| {
            let (px, py) = camera.project(x as f32, y as f32, z as f32);
            let p = (px, py, z as f32);
            [p, p]
        })
        .collect()
}

/// Camera pose; the same yaw/pitch/perspective pipeline as the original
/// textplots sketch in agent-rt/src/tools/landscape.rs.
struct Camera {
    pitch_rad: f32,
    yaw_rad: f32,
    distance: f32,
}

impl Camera {
    fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let (cp, sp) = (self.pitch_rad.cos(), self.pitch_rad.sin());
        let (cy, sy) = (self.yaw_rad.cos(), self.yaw_rad.sin());

        // Rotate: yaw around Z, then pitch around X.
        let x_rot = x * cy - y * sy;
        let y_temp = x * sy + y * cy;
        let y_rot = y_temp * cp - z * sp;
        let z_rot = y_temp * sp + z * cp;

        // Perspective division.
        let z_final = z_rot + self.distance;
        (x_rot / z_final, y_rot / z_final)
    }
}

struct Args {
    expr: Expr,
    expr_src: String,
    camera: Camera,
    grid: usize,
    width: usize,
    height: usize,
}

fn parse_args(arguments_json: &str) -> Result<Args, String> {
    let v: serde_json::Value = serde_json::from_str(arguments_json)
        .map_err(|e| format!("arguments must be a JSON object: {e}"))?;
    if !v.is_object() {
        return Err("arguments must be a JSON object".into());
    }

    let num = |key: &str, default: f64, min: f64, max: f64| -> f64 {
        v.get(key)
            .and_then(|n| n.as_f64())
            .unwrap_or(default)
            .clamp(min, max)
    };
    let uint = |key: &str, default: i64, min: i64, max: i64| -> usize {
        v.get(key)
            .and_then(|n| n.as_i64())
            .unwrap_or(default)
            .clamp(min, max) as usize
    };

    let expr_src = v
        .get("expr")
        .and_then(|s| s.as_str())
        .unwrap_or(DEFAULT_EXPR);
    let expr = Expr::parse(expr_src)
        .map_err(|e| format!("invalid expr '{expr_src}': {e}"))?;

    // Keep the aspect ratio while shrinking into the cell budget.
    let (mut width, mut height) = (uint("width", 120, 8, 400), uint("height", 60, 4, 200));
    if width * height > MAX_CELLS {
        let scale = (MAX_CELLS as f64 / (width * height) as f64).sqrt();
        width = ((width as f64 * scale).floor() as usize).max(8);
        height = ((height as f64 * scale).floor() as usize).max(4);
    }

    Ok(Args {
        expr,
        expr_src: expr_src.to_string(),
        camera: Camera {
            pitch_rad: (num("pitch", 40.0, -89.0, 89.0) as f32).to_radians(),
            yaw_rad: (num("yaw", 35.0, -360.0, 360.0) as f32).to_radians(),
            distance: num("distance", 3.5, 0.1, 100.0) as f32,
        },
        grid: uint("grid", 15, 4, 200),
        width,
        height,
    })
}

/// Render and return a `braille-v1` document. Two modes:
/// - heightfield (`z = f(x, y)`, no `z` in the expression): a wireframe
///   mesh over [-2, 2]²;
/// - implicit surface (`f(x, y, z) = 0`, the expression uses `z`): the
///   zero-crossings over [-2, 2]³ as points, colored by world z — e.g.
///   a torus `(sqrt(x^2+y^2)-1.2)^2 + z^2 - 0.5^2`.
///
/// ```json
/// {"render": "braille-v1", "title": "f(x,y) expr", "width": 120,
///  "rows":   ["⠀⣠⠔…", …],   // braille, one string per text row
///  "levels": ["..12…", …],   // per cell: '0'-'9' height level, '.' empty
///  "z_range": [-0.4, 0.4]}   // surface values that levels 0 and 9 map to
/// ```
///
/// Viewers fit the dots to their own width and color cells by level.
/// KEEP IN SYNC with the parser in crates/tui/src/tool_panel.rs.
pub fn run(arguments_json: &str) -> Result<String, String> {
    let args = parse_args(arguments_json)?;

    let segments = if args.expr.uses_z() {
        // Implicit-surface mode: draw the zero-crossings of f(x,y,z)
        // over [-2, 2]^3 as projected points (zero-length segments).
        iso_points(&args.expr, &args.camera, args.grid)
    } else {
        heightfield_segments(&args.expr, &args.camera, args.grid)
    };

    let raster = braille::rasterize(&segments, args.width, args.height);
    let (z_min, z_max) = segments.iter().flatten().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(lo, hi), p| (lo.min(p.2), hi.max(p.2)),
    );
    let doc = serde_json::json!({
        "render": "braille-v1",
        "title": args.expr_src,
        "width": raster.width,
        "rows": raster.rows,
        "levels": raster.levels,
        "z_range": [z_min, z_max],
    });
    Ok(doc.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(args: &str) -> serde_json::Value {
        serde_json::from_str(&run(args).expect("args should work")).expect("valid JSON")
    }

    #[test]
    fn default_arguments_render_ripple() {
        let d = doc("{}");
        assert_eq!(d["render"], "braille-v1");
        assert_eq!(d["title"], DEFAULT_EXPR);
        let rows = d["rows"].as_array().unwrap();
        assert!(rows.len() > 5, "expected a non-trivial chart");
        assert_eq!(rows.len(), d["levels"].as_array().unwrap().len());
        assert!(rows.iter().any(|r| r.as_str().unwrap().chars().any(|c| c != '\u{2800}')));
    }

    #[test]
    fn custom_expression_and_size() {
        let d = doc(r#"{"expr": "sin(x)*cos(y)", "width": 40, "height": 16, "grid": 10}"#);
        assert_eq!(d["width"], 40);
        assert!(d["rows"].as_array().unwrap().len() <= 16);
        let z = d["z_range"].as_array().unwrap();
        assert!(z[0].as_f64().unwrap() < z[1].as_f64().unwrap());
    }

    /// The largest request must still fit a gossipsub message (64 KiB),
    /// including JSON-in-JSON escaping in the chat message envelope.
    #[test]
    fn output_fits_a_chat_message() {
        let out = run(r#"{"width": 400, "height": 200, "grid": 200}"#).unwrap();
        let d: serde_json::Value = serde_json::from_str(&out).unwrap();
        let w = d["width"].as_u64().unwrap() as usize;
        assert!(w * d["rows"].as_array().unwrap().len() <= MAX_CELLS);
        let envelope = serde_json::to_string(&serde_json::json!({ "content": out })).unwrap();
        assert!(envelope.len() < 48 * 1024, "{} bytes", envelope.len());
    }

    #[test]
    fn implicit_torus_renders_points_colored_by_z() {
        let d = doc(
            r#"{"expr": "(sqrt(x^2+y^2)-1.2)^2 + z^2 - 0.5^2", "width": 60, "height": 30, "grid": 30}"#,
        );
        assert_eq!(d["render"], "braille-v1");
        let rows = d["rows"].as_array().unwrap();
        let drawn: usize = rows
            .iter()
            .flat_map(|r| r.as_str().unwrap().chars())
            .filter(|c| *c != '\u{2800}')
            .count();
        assert!(drawn > 60, "a torus should give plenty of surface dots, got {drawn}");
        // Levels color by world z: both low and high rings present.
        let levels: String = d["levels"].as_array().unwrap().iter().flat_map(|l| l.as_str().unwrap().chars()).collect();
        assert!(levels.contains('0') || levels.contains('1'), "low ring colored: {levels}");
        assert!(levels.contains('8') || levels.contains('9'), "high ring colored: {levels}");
        let z = d["z_range"].as_array().unwrap();
        assert!(z[0].as_f64().unwrap() < z[1].as_f64().unwrap());
    }

    #[test]
    fn heightfield_mode_unchanged_without_z() {
        let d = doc(r#"{"expr": "cos(x)*sin(y)", "width": 40, "height": 16, "grid": 12}"#);
        let rows = d["rows"].as_array().unwrap();
        let drawn: usize = rows
            .iter()
            .flat_map(|r| r.as_str().unwrap().chars())
            .filter(|c| *c != '\u{2800}')
            .count();
        assert!(drawn > 20, "egg-crate wireframe renders, got {drawn}");
    }

    #[test]
    fn bad_expression_is_an_error() {
        assert!(run(r#"{"expr": "x +"}"#).unwrap_err().contains("invalid expr"));
    }

    #[test]
    fn bad_json_is_an_error() {
        assert!(run("not json").is_err());
    }
}
