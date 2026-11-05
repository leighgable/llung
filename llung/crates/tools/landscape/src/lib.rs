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
    "pitch": {"type": "number", "default": 40, "description": "Camera pitch in degrees (looking down)"},
    "yaw": {"type": "number", "default": 35, "description": "Camera yaw in degrees"},
    "distance": {"type": "number", "default": 3.5, "description": "Camera distance (must be > 0)"},
    "grid": {"type": "integer", "default": 15, "description": "Mesh resolution, 4..=200"},
    "width": {"type": "integer", "default": 120, "description": "Output width in characters, 8..=400 (width*height is capped at 10000 cells)"},
    "height": {"type": "integer", "default": 60, "description": "Output height in characters, 4..=200 (width*height is capped at 10000 cells)"}
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

/// Render and return a `braille-v1` document:
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
    let n = args.grid;

    // Sample the expression over [-2, 2]^2 and project each vertex.
    let mut mesh = vec![vec![(0.0f32, 0.0f32, 0.0f32); n]; n];
    for i in 0..n {
        for j in 0..n {
            let x = (i as f64 / (n - 1) as f64) * 4.0 - 2.0;
            let y = (j as f64 / (n - 1) as f64) * 4.0 - 2.0;
            let z = args.expr.eval(x, y);
            // Keep poles/asymptotes from swallowing the view.
            let z = if z.is_finite() { z.clamp(-10.0, 10.0) } else { 0.0 };
            let (px, py) = args.camera.project(x as f32, y as f32, z as f32);
            mesh[i][j] = (px, py, z as f32);
        }
    }

    // Wireframe edges: horizontal and vertical neighbors.
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

    let raster = braille::rasterize(&segments, args.width, args.height);
    let (z_min, z_max) = mesh
        .iter()
        .flatten()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| (lo.min(p.2), hi.max(p.2)));
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
    fn bad_expression_is_an_error() {
        assert!(run(r#"{"expr": "x +"}"#).unwrap_err().contains("invalid expr"));
    }

    #[test]
    fn bad_json_is_an_error() {
        assert!(run("not json").is_err());
    }
}
