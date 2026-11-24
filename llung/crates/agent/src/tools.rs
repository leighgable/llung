// crates/agent/src/tools.rs
//
// Host-side tool catalog. Each wasm component in `runtime-tools/` also
// advertises the same metadata via its `get-metadata` export — this
// static copy exists so the orchestrator can put the catalog in the
// system prompt *before* deciding to load a component. Keep the two in
// sync when adding or changing a tool.
//
// Host-native tools (`host/` — glob, grep, edit) run in-process with no
// external processes; their specs live here only.

pub mod host;

/// One line of the system-prompt tool catalog.
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON schema for `arguments` (kept for host-side validation; the
    /// prompt uses the compact `usage` line instead).
    pub schema: &'static str,
    /// One-line argument summary shown to the model.
    pub usage: &'static str,
}

pub const IMAGE_RESIZE_TOOL_SCHEMA: &str = r#"
{
    "name": "image_mask_resize",
    "description": "Resize and mask an image as a thumbnail",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "size": {"type": "integer", "default": 128},
            "mask": {"type": "boolean", "default": true},
            "radius": {"type": "integer", "default": 20}
        },
        "required": ["path"]
    }
}
"#;

pub const LANDSCAPE_TOOL_SCHEMA: &str = r#"
{
    "name": "landscape",
    "description": "Render a 3D wireframe surface of a math expression f(x,y) as terminal braille art. Use when asked to plot, chart, graph, or visualize a function or landscape.",
    "parameters": {
        "type": "object",
        "properties": {
            "expr": {
                "type": "string",
                "default": "cos(sqrt(x^2+y^2)*3)*0.4",
                "description": "Surface expression. Without z: heightfield z = f(x,y) over [-2,2]^2. WITH z: implicit surface f(x,y,z) = 0 over [-2,2]^3, drawn as points (torus: '(sqrt(x^2+y^2)-1.2)^2 + z^2 - 0.5^2'). Variables: x, y, z. Operators: + - * / % ^. Functions: sin cos tan asin acos atan atan2 sqrt abs exp ln log2 log10 floor ceil round min max. Constants: pi tau e."
            },
            "pitch": {"type": "number", "default": 40, "description": "OPTIONAL camera pitch in degrees (looking down). Omit unless the user asks to adjust the view; values near 0 look edge-on and unreadable."},
            "yaw": {"type": "number", "default": 35, "description": "OPTIONAL camera yaw in degrees. Omit unless asked."},
            "distance": {"type": "number", "default": 3.5, "description": "OPTIONAL camera distance (must be > 0). Omit unless asked; values near 1 give extreme distortion."},
            "grid": {"type": "integer", "default": 15, "description": "OPTIONAL mesh resolution: 4..=200 heightfield; 4..=64 implicit (z) mode, PER AXIS — keep 25-40. Omit unless asked."},
            "width": {"type": "integer", "default": 120, "description": "OPTIONAL output width in characters, 8..=400 (width*height is capped at 10000 cells). Omit unless asked."},
            "height": {"type": "integer", "default": 60, "description": "OPTIONAL output height in characters, 4..=200 (width*height is capped at 10000 cells). Omit unless asked."}
        }
    }
}
"#;

pub const ZOOM_TOOL_SCHEMA: &str = r#"
{
    "name": "zoom",
    "description": "Expand a compressed part of the chat history back to its full text. Use when a one-line summary references details you need to answer well.",
    "parameters": {
        "type": "object",
        "properties": {
            "start": {"type": "integer", "description": "First history index to expand — the [n] or [a-b] number shown on the summary line."},
            "end": {"type": "integer", "description": "One past the last index to expand (so start..end covers the lines you need)."}
        },
        "required": ["start", "end"]
    }
}
"#;

pub const SUGI_TOOL_SCHEMA: &str = r#"
{
    "name": "sugi",
    "description": "Draw a directed graph (flowchart, pipeline, architecture) as a layered box diagram. Plain JSON in — nodes and edges — automatic layered layout. Use when asked to diagram, flowchart, or draw a system, process, or architecture. Prefer over 'diagram' unless nested boxes are needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "nodes": {"type": "array", "description": "Nodes: strings ['A', ...] or objects [{'id': 'A', 'label': 'Human text'}, ...]. Labels default to the id."},
            "edges": {"type": "array", "description": "Edges as pairs [['A','B'], ...] or objects [{'from': 'A', 'to': 'B', 'label': 'optional'}, ...]."},
            "title": {"type": "string", "description": "OPTIONAL short title shown next to the panel. Omit unless useful."},
            "width": {"type": "integer", "description": "OPTIONAL maximum width in characters, 8..=200. Omit unless asked."}
        },
        "required": ["nodes", "edges"]
    }
}
"#;

pub const DIAGRAM_TOOL_SCHEMA: &str = r#"
{
    "name": "diagram",
    "description": "Draw a flowchart or nested-box diagram as terminal box-drawing art. Use when asked to diagram, flowchart, or draw boxes-and-arrows of a system, process, or architecture.",
    "parameters": {
        "type": "object",
        "properties": {
            "graph": {"type": "string", "description": "The diagram in diagram-language, one statement per line: 'ID: Label' declares a node; 'box ID: Label { ... }' declares a nested container (close with '}'); 'A -> B: optional label' declares edges (chains A -> B -> C work; edges may name a box to connect to its first/last inner node); '#' starts a comment."},
            "title": {"type": "string", "description": "OPTIONAL short title shown next to the panel. Omit unless useful."},
            "width": {"type": "integer", "default": 64, "description": "Maximum width in characters, 24..=110. Smaller fits narrow chat panels."}
        },
        "required": ["graph"]
    }
}
"#;

pub const PLOT_DEVICE_TOOL_SCHEMA: &str = r#"
{
    "name": "plot_device",
    "description": "Plot tabular data (CSV) as terminal braille: scatter plots (2D, or 3D with a z column), histograms, and CDFs. Use when asked to plot, chart, or visualize data, columns, or a CSV file.",
    "parameters": {
        "type": "object",
        "properties": {
            "csv": {"type": "string", "description": "The CSV data itself. Omit when 'path' is given — the chat host reads the file and injects the content."},
            "path": {"type": "string", "description": "Path to a CSV file in the workspace or an operator data directory (LLUNG_RO_BINDS); the chat host reads it and injects the content as 'csv'. Use this instead of pasting data. Other paths are refused."},
            "kind": {"type": "string", "default": "scatter", "description": "Plot kind: scatter (default; 3D when 'z' is given), line, histogram, or cdf."},
            "x": {"type": "string", "description": "Domain column (scatter/line): header NAME or 0-based index. OMIT to use the row index (0,1,2,…) — the usual choice for a series over time. For histogram/cdf, x is the VALUE column."},
            "y": {"type": "string", "default": "1", "description": "Value column (scatter/line): name or index."},
            "z": {"type": "string", "description": "OPTIONAL column for z — 3D scatter with a pitch/yaw/distance camera. Omit for 2D."},
            "bins": {"type": "integer", "description": "Histogram bin count, 4..=200. Omit for one bin per data point (a bar per row)."},
            "width": {"type": "integer", "default": 120, "description": "Output width in characters, 8..=400 (width*height is capped at 10000 cells)"},
            "height": {"type": "integer", "default": 30, "description": "Output height in characters, 4..=200; defaults to a letterbox 30 (width*height is capped at 10000 cells)"}
        }
    }
}
"#;

pub const GLOB_TOOL_SCHEMA: &str = r#"
{
    "name": "glob",
    "description": "Find files by glob pattern, automatically respecting .gitignore and hidden-file rules. Returns absolute paths. Use to locate files before reading or editing them.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.rs' matches Rust files at any depth; '*.rs' also works."},
            "path": {"type": "string", "description": "OPTIONAL root directory (default: current workspace). Omit unless asked."}
        },
        "required": ["pattern"]
    }
}
"#;

pub const GREP_TOOL_SCHEMA: &str = r#"
{
    "name": "grep",
    "description": "Search file contents with a regex, respecting .gitignore. Use to find where a symbol, string, or pattern appears in the codebase.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern (Rust regex syntax)."},
            "path": {"type": "string", "description": "OPTIONAL file or directory to search (default: current workspace). Omit unless asked."},
            "include": {"type": "string", "description": "OPTIONAL file-name glob filter, e.g. '*.rs'. Omit to search all files."}
        },
        "required": ["pattern"]
    }
}
"#;

pub const EDIT_TOOL_SCHEMA: &str = r#"
{
    "name": "edit",
    "description": "Edit a text file by exact search-and-replace. Each old string must match EXACTLY ONCE in the file (include surrounding context to make it unique). Returns a unified diff of the change.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to edit."},
            "edits": {"type": "array", "items": {"type": "object", "properties": {"old": {"type": "string"}, "new": {"type": "string"}}, "required": ["old", "new"]}, "description": "Ordered exact replacements; every 'old' must match exactly once."}
        },
        "required": ["path", "edits"]
    }
}
"#;

pub const BASH_TOOL_SCHEMA: &str = r#"
{
    "name": "bash",
    "description": "Run a shell command in an isolated sandbox (bubblewrap) with a read-only workspace, memory limit, and timeout. The sandbox sees ONLY the workspace plus operator data directories (LLUNG_RO_BINDS, read-only at their own paths). Use for builds, tests, and any computation the file tools can't do.",
    "parameters": {
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "Shell command to run (sh -c)."},
            "timeout_secs": {"type": "integer", "description": "OPTIONAL wall-clock limit in seconds (default 30, max 300). Omit unless the command needs long."},
            "writable": {"type": "boolean", "description": "OPTIONAL mount the workspace writable (default false = read-only). Writing needs approval; expect denial."},
            "memory_mb": {"type": "integer", "description": "OPTIONAL memory limit in MB (default 512). Omit unless asked."}
        },
        "required": ["cmd"]
    }
}
"#;

pub const REPL_TOOL_SCHEMA: &str = r#"
{
    "name": "repl",
    "description": "Evaluate code in a stateful REPL session (python, chibi-scheme, futhark) that persists across calls. Use for multi-step calculations, prototyping, or data exploration. (torch lives behind the gpu:torch tool — it needs a GPU grant.)",
    "parameters": {
        "type": "object",
        "properties": {
            "interp": {"type": "string", "description": "python | chibi-scheme | futhark | torch (torch only via the gpu:torch tool)"},
            "input": {"type": "string", "description": "Code to evaluate this turn."},
            "session": {"type": "string", "description": "OPTIONAL session name separating independent states (default: the interpreter name). Use it to keep separate calculations apart."},
            "timeout_secs": {"type": "integer", "description": "OPTIONAL per-evaluation wall-clock limit (default 30, max 120)."},
            "backend": {"type": "string", "description": "Futhark only: c (default). opencl/cuda need approval — expect denial."}
        },
        "required": ["interp", "input"]
    }
}
"#;

/// The torch trainer REPL: python with the ROCm torch closure, inside
/// the GPU-enabled sandbox. Tier::Gpu — the operator must grant
/// `gpu:torch` (once, or `/grant gpu:torch perpetual`) before a call
/// runs; the plain `repl` tool refuses this interpreter.
pub const GPU_TORCH_TOOL_SCHEMA: &str = r#"
{
    "name": "gpu:torch",
    "description": "Evaluate code in a stateful PyTorch REPL session with GPU (ROCm) access that persists across calls. Use for tensor math, model prototyping, and ML experiments. Requires the gpu:torch grant — the operator may need to approve the first call.",
    "parameters": {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Python/torch code to evaluate this turn."},
            "session": {"type": "string", "description": "OPTIONAL session name separating independent states (default: torch). Use it to keep separate experiments apart."},
            "timeout_secs": {"type": "integer", "description": "OPTIONAL per-evaluation wall-clock limit (default 30, max 120)."}
        },
        "required": ["input"]
    }
}
"#;

pub const SHOW_IMAGE_TOOL_SCHEMA: &str = r#"
{
    "name": "show_image",
    "description": "Publish an image file (plot, figure) from the workspace to the chat as an attachment. Use after saving a matplotlib PNG (plt.savefig); do NOT paste image data into chat.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Image file path, relative to the workspace root. Paths escaping the workspace (..) are rejected."},
            "caption": {"type": "string", "description": "OPTIONAL short caption shown under the image."}
        },
        "required": ["path"]
    }
}
"#;

pub const ARTIST_TOOL_SCHEMA: &str = r#"
{
    "name": "artist",
    "description": "Generate procedural art from a seed as colored terminal braille. Use when asked to draw, paint, or make generative art, a pattern, or a picture.",
    "parameters": {
        "type": "object",
        "properties": {
            "seed": {"type": "integer", "default": 42, "description": "Art seed. The same seed always produces the same artwork; change it for a new piece."},
            "style": {"type": "string", "default": "mix", "description": "Art style: clouds (soft billowing), ridged (sharp crests), rings (warped ripples), weave (woven interference), or mix (pick per seed)."},
            "zoom": {"type": "number", "default": 3, "description": "Pattern scale (must be > 0); larger zooms into the noise."},
            "width": {"type": "integer", "default": 120, "description": "Output width in characters, 8..=400 (width*height is capped at 10000 cells)"},
            "height": {"type": "integer", "default": 30, "description": "Output height in characters, 4..=200; defaults to a letterbox 30 (width*height is capped at 10000 cells)"}
        }
    }
}
"#;

/// All tools available to the agent, host-side copies.
pub fn tool_catalog() -> &'static [ToolSpec] {
    &[
        ToolSpec {
            name: "image_mask_resize",
            description: "Resize and mask an image as a thumbnail",
            schema: IMAGE_RESIZE_TOOL_SCHEMA,
            usage: "image_mask_resize: resize/mask an image thumbnail. args: path, size, mask, radius.",
        },
        ToolSpec {
            name: "zoom",
            description: "Expand a compressed history summary back to full text",
            schema: ZOOM_TOOL_SCHEMA,
            usage: "zoom: load the FULL text of compressed history lines into your \
                     context. History lines are numbered like [3] or [2-5]; pass start \
                     and end (end one past the last line) to expand that range. args: \
                     start, end.",
        },
        ToolSpec {
            name: "plot_device",
            description: "Plot CSV data as braille scatter (2D/3D), histogram, or CDF",
            schema: PLOT_DEVICE_TOOL_SCHEMA,
            usage: "plot_device: plot tabular data (CSV). Pass a CSV FILE by 'path' (the host \
                     reads it — never paste data) or 'csv' for small inline data. args: path \
                     (or csv), kind (scatter|line|histogram|cdf), y (value column; x is the \
                     domain, OMIT it to use the row index — perfect for a series over time), \
                     z (3D scatter), bins (histogram; omit = one bar per row), width, height \
                     (default 120x30 letterbox).",
        },
        ToolSpec {
            name: "sugi",
            description: "Draw a directed graph as a layered box diagram from plain JSON nodes/edges (Sugiyama layout)",
            schema: SUGI_TOOL_SCHEMA,
            usage: "sugi: flowchart/pipeline/architecture from plain JSON. args: nodes \
                     (required: ['A', ...] or [{'id':'A','label':'text'}, ...]), edges \
                     (required: [['A','B'], ...] or [{'from':'A','to':'B','label':'...'}]), \
                     title, width (optional). Keep node labels short (1-3 words); ids must \
                     match edges exactly.",
        },
        ToolSpec {
            name: "diagram",
            description: "Draw a flowchart or nested-box diagram as terminal box-drawing art (use for diagram/flowchart/architecture requests)",
            schema: DIAGRAM_TOOL_SCHEMA,
            usage: "diagram: boxes-and-arrows / flowchart / nested-box art. Pass ONE 'graph' \
                     string, one statement per line: 'ID: Label' = node; 'box ID: Label { ... \
                     }' = nested container (close with '}'); 'A -> B: label' = edge (chains \
                     work; may name a box); '#' = comment. args: graph, title, width. \
                     Example flowchart graph: 'INTAKE: Intake\nbox INVESTIGATE: Investigate {\n  \
                     SUBA: Sub Agent A\n  SUBB: Sub Agent B\n}\nINTAKE -> INVESTIGATE\nSUBA -> \
                     PLAN\nSUBB -> PLAN\nPLAN: Draft Plan\nPLAN -> REPORT\nREPORT: Report'.",
        },
        ToolSpec {
            name: "artist",
            description: "Generate procedural art from a seed as colored terminal braille (use for draw/paint/generative-art requests)",
            schema: ARTIST_TOOL_SCHEMA,
            usage: "artist: procedural art from a seed. args: seed (integer; same seed = same \
                     artwork), style (clouds|ridged|rings|weave|mix — mix picks per seed), zoom, \
                     width, height.",
        },
        ToolSpec {
            name: "glob",
            description: "Find files by glob pattern (gitignore-aware)",
            schema: GLOB_TOOL_SCHEMA,
            usage: "glob: find files by pattern. args: pattern (required, e.g. '**/*.rs' — \
                     matches at any depth), path (optional root dir, default workspace). \
                     Respects .gitignore and skips hidden files.",
        },
        ToolSpec {
            name: "grep",
            description: "Search file contents with a regex (gitignore-aware)",
            schema: GREP_TOOL_SCHEMA,
            usage: "grep: search file contents with a regex. Returns matching lines as \
                     path:line:text. args: pattern (required), path (optional file or dir, \
                     default workspace), include (optional glob filter like '*.rs').",
        },
        ToolSpec {
            name: "edit",
            description: "Edit a text file by exact search-and-replace (returns a unified diff)",
            schema: EDIT_TOOL_SCHEMA,
            usage: "edit: exact search-and-replace in a text file. args: path (required), \
                     edits: array of {old, new} applied in order. Every 'old' must match \
                     EXACTLY ONCE — include enough surrounding context lines to make it \
                     unique, or the edit is rejected with no changes written. Returns a \
                     unified diff. Never invent file content; if unsure, glob/grep first.",
        },
        ToolSpec {
            name: "repl",
            description: "Stateful sandboxed REPL sessions (python, chibi-scheme, futhark) that persist across calls",
            schema: REPL_TOOL_SCHEMA,
            usage: "repl: evaluate code in a persistent interpreter session — variables and \
                     definitions survive between calls. args: interp (python|chibi-scheme|futhark), \
                     input (code to run this turn), session (optional name; separate names = \
                     separate state), timeout_secs (default 30), backend (futhark only; c is \
                     the default — opencl/cuda need approval). For torch use the gpu:torch tool.",
        },
        ToolSpec {
            name: "gpu:torch",
            description: "Stateful PyTorch REPL session with GPU access (needs the gpu:torch grant)",
            schema: GPU_TORCH_TOOL_SCHEMA,
            usage: "gpu:torch: evaluate torch code in a persistent GPU-backed python session — \
                     variables, tensors and models survive between calls (sessions live inside \
                     the GPU sandbox; the operator grants gpu:torch, once or via \
                     `/grant gpu:torch perpetual`). args: input (python/torch code to run this \
                     turn), session (optional name; default 'torch'), timeout_secs (default 30).",
        },
        ToolSpec {
            name: "show_image",
            description: "Publish a workspace image (matplotlib PNG plot) to the chat as an attachment",
            schema: SHOW_IMAGE_TOOL_SCHEMA,
            usage: "show_image: publish an image file from the workspace as a chat attachment. \
                     Save the plot first inside a gpu:torch session (matplotlib Agg backend, \
                     e.g. plt.savefig('experiments/runs/exp07/loss.png')), then call show_image \
                     with that path. args: path (required, inside the workspace), caption \
                     (optional). Never paste image data into chat.",
        },
        ToolSpec {
            name: "bash",
            description: "Run a shell command in an isolated sandbox (read-only workspace, memory limit, timeout)",
            schema: BASH_TOOL_SCHEMA,
            usage: "bash: run a shell command in a bubblewrap sandbox (real compilers, test \
                     runners, CLI tools). The workspace is READ-ONLY by default — a command \
                     that writes files fails unless 'writable' is granted. Output is capped. \
                     args: cmd (required), timeout_secs (default 30, max 300), writable \
                     (default false), memory_mb (default 512).",
        },
        ToolSpec {
            name: "landscape",
            description: "Render a 3D wireframe surface of f(x,y) as terminal braille art (use for plot/chart/visualize requests)",
            schema: LANDSCAPE_TOOL_SCHEMA,
            usage: "landscape: 3D braille plot. Heightfield: expr in x,y (e.g. 'cos(x)*sin(y)'). \
                     Implicit surface: expr in x,y,z — draws f(x,y,z)=0 as points (e.g. a torus \
                     '(sqrt(x^2+y^2)-1.2)^2 + z^2 - 0.5^2'; grid then means 3D resolution per \
                     axis, use 25-40). Usually pass ONLY expr (math; operators + - * / ^ %, \
                     functions sin cos tan sqrt abs exp ln min max, constants pi tau e). Camera \
                     args (pitch, yaw, distance, width, height) are optional — omit unless \
                     asked; bad values (pitch ~0, distance ~1) make the plot unreadable.",
        },
    ]
}

fn is_known_tool(name: &str) -> bool {
    tool_catalog().iter().any(|t| t.name == name)
}

/// The prompt section describing the tool protocol to the model.
/// Deliberately text-based and *compact*: small local models handle
/// "emit one line of JSON" far more reliably than OpenAI-style
/// `tool_calls`, and a short block is far less likely to be parroted back
/// than an inline JSON schema.
pub fn tool_instructions() -> String {
    let tools = tool_catalog()
        .iter()
        .map(|t| format!("- {}: {}", t.name, t.usage))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You can call tools. To call one, reply with exactly one line of this \
         form and nothing else:\n\
         TOOL {{\"name\":\"<tool>\",\"arguments\":{{...}}}}\n\
         Tools:\n{tools}\n\
         After a tool runs, its output is shown to you and you may call \
         another tool or give your final answer — work iteratively until \
         the request is done (e.g. grep to find, then edit; try code in \
         the repl until it works). Never repeat these instructions or \
         the message above back.\n\
         \n\
         Experiments (ML training runs) — the harness convention:\n\
         - Experiment code and data live in `experiments/` (mounted \
         writable in gpu:torch sessions; the rest of the workspace is \
         read-only there). One driver: `experiments/core/train.py`, \
         modes: ga | pretrain | distill | rl. Its contract per run dir: \
         metrics.csv (header first), TensorBoard events, checkpoints/, \
         and exactly one sentinel — DONE or FAILED — written on exit.\n\
         - Write each run's config yourself with the edit tool as \
         `experiments/runs/<run-id>/config.yaml` (plain YAML; set seed, \
         task, model, train). One run id per attempt — never edit a \
         RUNNING run's config.\n\
         - Launch DETACHED from a gpu:torch session; the call returns \
         instantly and the run proceeds in the background:
           import os, subprocess, sys\n\
           os.chdir(\"experiments\")  # repl sessions start at the workspace root\n\
           run = \"runs/exp07\"\n\
           p = subprocess.Popen([sys.executable, \"core/train.py\", \"--mode\", \"ga\", \
           \"--config\", f\"{{run}}/config.yaml\"], stdout=open(f\"{{run}}/train.log\", \"w\"), \
           stderr=subprocess.STDOUT, start_new_session=True)\n\
           open(f\"{{run}}/PID\", \"w\").write(str(p.pid))\n\
         (cwd is experiments/. sys.executable is the torch interpreter.)\n\
         - Need a new python package? Edit pyproject.toml dependencies, \
           then run bash (writable=true): uv lock — the torch \
           interpreter env rebuilds with it. (uv and git are available \
           inside the sandbox.)
\
         - The trainer has a property suite: run \
           subprocess.run([sys.executable, '-m', 'pytest', 'core/tests', \
           '-q']) from a gpu:torch session after changing core/train.py \
           or core/monitor.py — properties (GA elitism, seeded \
           reproducibility, sentinels) must hold for every seed.
\
         - External data (e.g. teacher traces on a drive) is available \
           read-only under its own absolute path when the operator set \
           LLUNG_RO_BINDS — stream shards from there; write only into \
           the run dir. Never try to write to those paths.
\
         - Monitor by polling the run dir (idempotent — safe to repeat): \
           import os, sys; os.chdir(\"experiments\"); sys.path.insert(0, \"core\"); \
           import monitor; print(monitor.summarize(\"runs/exp07\")). DONE/FAILED \
           tell you a run ended; monitor.summarize prints the last metrics \
           rows plus alerts (plateaus, grad-norm spikes). For a quick \
           visual glance at metrics.csv curves use plot_device with the \
           csv path. For a REAL plot, save a PNG (matplotlib, \
           plt.savefig) from the gpu:torch session and publish it with \
           the show_image tool — never paste image data into chat."
    )
}

/// Cap for host-side file inlining into tool arguments.
const MAX_DATA_FILE_BYTES: usize = 256 * 1024;

/// Host-side data-file resolution: wasm tools have no I/O, so when a
/// `plot_device` call names a file by `path`, the HOST reads it and
/// injects the content as `csv` before the sandboxed call. The `path`
/// key is kept (panels show it) alongside the injected `csv`.
///
/// `confine_to`: `Some(workspace)` for AGENT calls — the read happens
/// outside any sandbox, so the path must pass
/// [`crate::paths::agent_readable`] (workspace + LLUNG_RO_BINDS; no
/// `..`/symlink escapes). `None` for the local user's own `/plot`.
pub async fn resolve_data_file(
    mut call: ToolCall,
    confine_to: Option<&std::path::Path>,
) -> Result<ToolCall, String> {
    if call.name != "plot_device" {
        return Ok(call);
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&call.arguments) else {
        return Ok(call); // malformed args: let the tool report it
    };
    if value.get("csv").is_some() || !value.get("path").is_some() {
        return Ok(call);
    }
    let Some(path) = value.get("path").and_then(|p| p.as_str()) else {
        return Ok(call);
    };
    let expanded = match confine_to {
        Some(workspace) => crate::paths::agent_readable(path, workspace)?,
        None => std::path::PathBuf::from(shellexpand::tilde(path).as_ref()),
    };
    let meta = std::fs::metadata(&expanded)
        .map_err(|e| format!("cannot read '{path}': {e}"))?;
    if meta.len() as usize > MAX_DATA_FILE_BYTES {
        return Err(format!(
            "'{path}' is {} bytes; the cap is {} — plot a smaller excerpt",
            meta.len(),
            MAX_DATA_FILE_BYTES
        ));
    }
    let content =
        std::fs::read_to_string(&expanded).map_err(|e| format!("cannot read '{path}': {e}"))?;
    let mut map = value.as_object().cloned().unwrap_or_default();
    map.insert("csv".into(), serde_json::Value::String(content));
    call.arguments = serde_json::Value::Object(map).to_string();
    Ok(call)
}

/// Run a tool directly (user-invoked `/plot` and friends): resolves data
/// files, executes the component through the wasm worker, and returns
/// the raw output. No aukora ceremony — this is the local user acting.
pub async fn run_tool_locally(name: &str, arguments_json: &str) -> Result<String, String> {
    if !is_known_tool(name) {
        return Err(format!("unknown tool '{name}' — try /tools"));
    }
    let call = ToolCall {
        name: name.to_string(),
        arguments: arguments_json.to_string(),
    };
    // The local user acting: no agent read confinement.
    let call = resolve_data_file(call, None).await?;
    // Host-native tools (glob, grep, edit) run in-process — no wasm file.
    if let Some(result) = host::dispatch(name, &call.arguments) {
        return result;
    }
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let tx = agent_rt::spawn_wasm_worker();
    // Same resolution as the secured path (aukora/security.rs).
    tx.send(agent_rt::ToolRequest {
        wasm_path: crate::paths::tool_wasm_path(name),
        arguments_json: call.arguments,
        response_tx: resp_tx,
    })
    .await
    .map_err(|e| format!("wasm worker unreachable: {e}"))?;
    resp_rx
        .await
        .map_err(|e| format!("wasm worker dropped: {e}"))?
}

/// True when the model parroted its own prompt instead of answering.
/// Publishing such text would feed the echo back into the compressed
/// history and make the next turn worse.
pub fn looks_like_echo(reply: &str) -> bool {
    reply.contains("New message:")
        || reply.contains("Available tools")
        || reply.contains("TOOLS:")
        || reply.contains("<tool>")
}

/// A tool call parsed out of a model reply.
#[derive(Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    /// JSON object, serialized.
    pub arguments: String,
}

/// Scan a reply for a `TOOL {...}` call. The marker may appear anywhere
/// in a line (small models prefix prose before it); the name must be in
/// the catalog, which also rejects echoed instruction templates like
/// `"<tool>"`. Anything else means "no tool call".
///
/// Two syntaxes are accepted — small models invent the second one:
///   TOOL {"name":"landscape","arguments":{"expr":"sin(x)"}}   (canonical)
///   TOOL landscape {"expr":"sin(x)"}                          (bare name)
pub fn parse_tool_call(reply: &str) -> Option<ToolCall> {
    for line in reply.lines() {
        let mut search = line;
        while let Some(pos) = search.find("TOOL") {
            let rest = search[pos + 4..].trim_start();
            let (bare_name, args_at) = leading_ident(rest);
            if let Some(json) = extract_json_object(args_at) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(json) {
                    if let Some(call) = tool_call_from_value(value, bare_name) {
                        return Some(call);
                    }
                }
            }
            search = &search[pos + 4..];
        }
    }
    None
}

/// A word immediately after the TOOL marker (`TOOL landscape {...}`),
/// or None when the marker is followed directly by JSON. Bare words in
/// running prose ("toolbox") are not names: the next character must be
/// whitespace or the arguments object.
fn leading_ident(s: &str) -> (Option<&str>, &str) {
    let end = s
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))
        .unwrap_or(s.len());
    if end == 0 {
        return (None, s);
    }
    match s[end..].chars().next() {
        Some(c) if c.is_whitespace() || c == '{' => (Some(&s[..end]), s[end..].trim_start()),
        _ => (None, s),
    }
}

/// First balanced `{...}` in `s` (string-aware), if any. Models often
/// append prose after the JSON, which `serde_json::from_str` would reject.
fn extract_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in s.char_indices().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

fn tool_call_from_value(
    value: serde_json::Value,
    bare_name: Option<&str>,
) -> Option<ToolCall> {
    // Canonical form: the object carries the name.
    if let Some(name) = value.get("name").and_then(|n| n.as_str()) {
        if !is_known_tool(name) {
            return None;
        }
        let arguments = value
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        if !arguments.is_object() {
            return None;
        }
        return Some(ToolCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    // Bare-name form: TOOL landscape {"expr": "sin(x)"} — the object
    // IS the arguments. (Guards: known name, plain object.)
    if let Some(name) = bare_name {
        if is_known_tool(name) && value.is_object() {
            return Some(ToolCall {
                name: name.to_string(),
                arguments: value.to_string(),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_line_with_arguments() {
        let call = parse_tool_call("Let me plot that.\nTOOL {\"name\": \"landscape\", \"arguments\": {\"expr\": \"sin(x)\"}}\nDone.")
            .expect("tool line should parse");
        assert_eq!(call.name, "landscape");
        assert!(call.arguments.contains("sin(x)"));
    }

    #[test]
    fn parses_tool_line_without_arguments() {
        let call = parse_tool_call("TOOL {\"name\": \"landscape\"}").expect("no-arg call");
        assert_eq!(call.name, "landscape");
        assert_eq!(call.arguments, "{}");
    }

    #[test]
    fn plain_text_is_not_a_tool_call() {
        assert!(parse_tool_call("Just a normal reply.").is_none());
        assert!(parse_tool_call("TOOL not json").is_none());
        // Unknown names — including echoed instruction templates — are rejected.
        assert!(parse_tool_call("TOOL {\"name\": \"<tool>\", \"arguments\": {}}").is_none());
        assert!(parse_tool_call("TOOL {\"arguments\": {}}").is_none()); // no name
        assert!(parse_tool_call("TOOL {\"name\": \"landscape\", \"arguments\": [1]}").is_none());
    }

    #[test]
    fn parses_the_bare_name_shorthand() {
        // The exact drift seen in the wild: the model wraps nothing and
        // puts the arguments object bare after the tool name.
        let call = parse_tool_call(
            "\nTOOL landscape {\"expr\":\"cos(0.1*x)*sin(0.1*y)\",\"grid\":true,\"pitch\":0}",
        )
        .expect("bare-name form should parse");
        assert_eq!(call.name, "landscape");
        assert!(call.arguments.contains("cos(0.1*x)*sin(0.1*y)"));
        assert!(call.arguments.contains("\"grid\":true"));

        // Canonical form still wins when both could apply.
        let call = parse_tool_call("TOOL landscape {\"name\":\"artist\",\"arguments\":{}}")
            .expect("name field wins");
        assert_eq!(call.name, "artist");

        // An unknown bare name is still rejected.
        assert!(parse_tool_call("TOOL cubist {\"a\":1}").is_none());
        // "TOOLBOX" in prose is not a marker+name.
        assert!(parse_tool_call("my TOOLBOX is red").is_none());
    }

    #[test]
    fn tool_marker_mid_line_is_found() {
        let call = parse_tool_call(
            "Sure! TOOL {\"name\":\"landscape\",\"arguments\":{\"expr\":\"x^2-y^2\"}} hope this helps",
        )
        .expect("mid-line marker should parse");
        assert_eq!(call.name, "landscape");
        assert!(call.arguments.contains("x^2-y^2"));
    }

    #[test]
    fn echoes_are_detected() {
        assert!(looks_like_echo(
            "New message: leroy: hi\nTOOL {\"name\": \"<tool>\"}"
        ));
        assert!(looks_like_echo("Available tools:\n- landscape"));
        assert!(!looks_like_echo("here is the surface you asked for"));
    }

    /// The shipped sug.wasm component executes end-to-end through the
    /// wasm worker, the same path the agent takes. Skips when the
    /// component isn't built (same pattern as the landscape test).
    #[tokio::test]
    async fn sugi_component_executes_end_to_end() {
        let wasm = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../runtime-tools/sugi.wasm");
        if !wasm.exists() {
            eprintln!("runtime-tools/sugi.wasm not built; skipping");
            return;
        }
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var(
                "LLUNG_TOOLS_DIR",
                wasm.parent().expect("wasm has a parent dir"),
            );
        }
        let out = run_tool_locally(
            "sugi",
            r#"{"nodes": [{"id": "A", "label": "Intake"}, "B", "C"], "edges": [["A", "B"], ["B", "C"]], "title": "flow"}"#,
        )
        .await
        .expect("sugi executes");
        let doc: serde_json::Value = serde_json::from_str(&out).expect("sugi-v1 JSON output");
        assert_eq!(doc["render"], "sugi-v1");
        assert_eq!(doc["title"], "flow");
        let ys: Vec<f64> = ["A", "B", "C"]
            .iter()
            .map(|id| {
                doc["nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|n| n["id"] == *id)
                    .unwrap()["y"]
                    .as_f64()
                    .unwrap()
            })
            .collect();
        assert!(ys[0] < ys[1] && ys[1] < ys[2], "layers flow down: {ys:?}");
    }

    #[test]
    fn instructions_list_every_tool() {
        let instructions = tool_instructions();
        for spec in tool_catalog() {
            assert!(instructions.contains(spec.name), "catalog missing {}", spec.name);
        }
        assert!(instructions.contains("TOOL {"));
    }
}
