# artist-tool

llung wasm tool: procedural generated art from a seed, drawn as colored
terminal braille directly in the chat.

Everything derives from an integer seed by avalanche hashing — no RNG
state, no I/O, no side effects, bounded cost. Same seed ⇒ same artwork
on every machine.

## Arguments

| field    | default | notes                                             |
|----------|---------|---------------------------------------------------|
| `seed`   | `42`    | the piece's identity; change for a new artwork    |
| `style`  | `mix`   | `clouds` · `ridged` · `rings` · `weave` · `mix`   |
| `zoom`   | `3`     | pattern scale (`> 0`)                             |
| `width`  | `120`   | chars, clamped to `8..=400`                       |
| `height` | `60`    | chars, clamped to `4..=200`                       |

Sizes are capped at `width * height <= 10000` cells, so the result
always fits one chat message (gossipsub's 64 KiB limit).

- **clouds** — two-stage domain-warped fBm (soft billowing)
- **ridged** — ridged multifractal (sharp crests)
- **rings** — concentric rings warped by noise (ripples)
- **weave** — trig interference with a noise wash (woven fabric)
- **mix** — picks one of the above deterministically per seed

The field is sampled per braille dot and thresholded against an ordered
8×8 Bayer matrix, so gradients render as stipple; each cell carries a
brightness level (0–9) that the viewer colors with its gradient.

## Output

A `braille-v1` render document (JSON) — the same format `landscape`
emits — so the TUI shows it as a pi-style tool panel, fitted to the
viewer's width and colored by level, with no viewer-side changes.

## Build

The component targets `wasm32-wasip2`; nightly rustc emits a component
directly for this target (see `crates/tools/landscape/README.md` for
the nix-sysroot workaround when `rustup target add` is unavailable):

```sh
# from the workspace root
cargo build -p artist-tool --release --target wasm32-wasip2
cp target/wasm32-wasip2/release/artist_tool.wasm runtime-tools/artist.wasm
```

Preview on the host (no wasm needed):

```sh
cargo run -p artist-tool --example show_art -- '{"style": "rings", "seed": 11}'
```
