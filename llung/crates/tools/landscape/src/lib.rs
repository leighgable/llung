//! `landscape` — a llung wasm tool.
//!
//! Renders a 3D wireframe surface of a math expression `f(x, y)` as
//! terminal braille art. The model controls the math and the camera via
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
    "width": {"type": "integer", "default": 120, "description": "Output width in characters, 8..=400"},
    "height": {"type": "integer", "default": 60, "description": "Output height in characters, 4..=200"}
  }
}"#;

const DEFAULT_EXPR: &str = "cos(sqrt(x^2+y^2)*3)*0.4";

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

    Ok(Args {
        expr,
        camera: Camera {
            pitch_rad: (num("pitch", 40.0, -89.0, 89.0) as f32).to_radians(),
            yaw_rad: (num("yaw", 35.0, -360.0, 360.0) as f32).to_radians(),
            distance: num("distance", 3.5, 0.1, 100.0) as f32,
        },
        grid: uint("grid", 15, 4, 200),
        width: uint("width", 120, 8, 400),
        height: uint("height", 60, 4, 200),
    })
}

pub fn run(arguments_json: &str) -> Result<String, String> {
    let args = parse_args(arguments_json)?;
    let n = args.grid;

    // Sample the expression over [-2, 2]^2 and project each vertex.
    let mut mesh = vec![vec![(0.0f32, 0.0f32); n]; n];
    for i in 0..n {
        for j in 0..n {
            let x = (i as f64 / (n - 1) as f64) * 4.0 - 2.0;
            let y = (j as f64 / (n - 1) as f64) * 4.0 - 2.0;
            let z = args.expr.eval(x, y);
            // Keep poles/asymptotes from swallowing the view.
            let z = if z.is_finite() { z.clamp(-10.0, 10.0) } else { 0.0 };
            mesh[i][j] = args.camera.project(x as f32, y as f32, z as f32);
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

    Ok(braille::rasterize(&segments, args.width, args.height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_arguments_render_ripple() {
        let out = run("{}").expect("default args should work");
        assert!(out.lines().count() > 5, "expected a non-trivial chart");
        assert!(out.chars().any(|c| ('\u{2800}'..='\u{28FF}').contains(&c)));
    }

    #[test]
    fn custom_expression_and_size() {
        let out = run(r#"{"expr": "sin(x)*cos(y)", "width": 40, "height": 16, "grid": 10}"#)
            .expect("custom args should work");
        assert!(out.lines().count() <= 16);
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
