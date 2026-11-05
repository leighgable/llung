# landscape-tool

llung wasm tool: renders a 3D wireframe surface of a math expression
`f(x, y)` as terminal braille art, directly in the chat.

The model controls the math and the camera through JSON arguments — the
expression evaluator (`src/expr.rs`) is a deliberately tiny language
(variables `x`/`y`, arithmetic, common functions), *not* a code sandbox:
no I/O, no side effects, bounded cost.

## Arguments

| field      | default                       | notes                                        |
|------------|-------------------------------|----------------------------------------------|
| `expr`     | `cos(sqrt(x^2+y^2)*3)*0.4`    | `f(x,y)`; vars `x y`, consts `pi tau e`      |
| `pitch`    | `40`                          | camera pitch, degrees                        |
| `yaw`      | `35`                          | camera yaw, degrees                          |
| `distance` | `3.5`                         | camera distance (`> 0`)                      |
| `grid`     | `15`                          | mesh resolution, clamped to `4..=200`        |
| `width`    | `120`                         | output width in chars, clamped to `8..=400`  |
| `height`   | `60`                          | output height in chars, clamped to `4..=200` |

Sizes are capped at `width * height <= 10000` cells (aspect kept), so the
result always fits one chat message (gossipsub's 64 KiB limit).

## Output

A `braille-v1` render document (JSON), which the TUI shows as a pi-style
tool panel: dots fitted to the viewer's width, colored by height with the
theme's elevation gradient, or drawn as an anti-aliased image in Kitty
graphics terminals (Ghostty, Kitty).

```json
{"render": "braille-v1", "title": "sin(x)*cos(y)", "width": 120,
 "rows":   ["⠀⣠⠔…", …],
 "levels": ["..12…", …],
 "z_range": [-1.0, 1.0]}
```

- `rows`: braille characters, one string per text row.
- `levels`: one char per cell, `'0'`-`'9'` = height level of the cell's
  highest dot (mapped linearly over `z_range`), `'.'` = empty cell.

The format is parsed in `crates/tui/src/tool_panel.rs`; keep them in sync.

## Build

The component targets `wasm32-wasip2`; nightly rustc emits a component
directly for this target (no `cargo component` needed).

```sh
# from the workspace root
rustup target add wasm32-wasip2            # if your toolchain supports it
cargo build -p landscape-tool --release --target wasm32-wasip2
cp target/wasm32-wasip2/release/landscape_tool.wasm runtime-tools/landscape.wasm
```

The nix-provided toolchain in this repo has no rustup; it ships x86_64
std only. Workaround used here: download the matching std component,
merge it into a writable sysroot, and build against that:

```sh
curl -LO https://static.rust-lang.org/dist/2026-07-20/rust-std-nightly-wasm32-wasip2.tar.xz
tar xf rust-std-nightly-wasm32-wasip2.tar.xz
mkdir -p ~/llung-rustc/lib/rustlib
cp -r <nix-rust>/lib/rustlib/x86_64-unknown-linux-gnu ~/llung-rustc/lib/rustlib/
cp -r rust-std-nightly-wasm32-wasip2/rust-std-wasm32-wasip2/lib/rustlib/wasm32-wasip2 ~/llung-rustc/lib/rustlib/
RUSTFLAGS="--sysroot=$HOME/llung-rustc" \
  cargo build -p landscape-tool --release --target wasm32-wasip2
cp target/wasm32-wasip2/release/landscape_tool.wasm runtime-tools/landscape.wasm
```

The host resolves tools from `runtime-tools/` next to the process cwd
(or `LLUNG_TOOLS_DIR`, if set).

## Try it

```sh
cargo run -p landscape-tool --example show_chart
```

## Keep in sync

`src/lib.rs` (`INPUT_SCHEMA` / `get_metadata`) is the source of truth the
host reads; `crates/agent/src/tools.rs` (`LANDSCAPE_TOOL_SCHEMA`) is the
copy shown to the LLM. Update both together, and note that
`wit/tool.wit` is duplicated from `crates/agent-rt/wit/tool.wit`.
