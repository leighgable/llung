# plot-device-tool

llung wasm tool: plot tabular data (CSV) as terminal braille art,
directly in the chat. Kinds: scatter (2D, or 3D with a `z` column and a
pitch/yaw/distance camera), histogram, and CDF. Output is a
`braille-v1` render document — the same format `landscape` emits, so
viewers color it by level with no new code.

The tool is pure compute: it receives the CSV as the `csv` argument.
The chat HOST resolves a `path` argument to the file's contents before
the sandboxed call (256 KiB cap), so models and users never paste data.

## Using it from the chat

```
# the agent plots a file you point at (pi-style mention expands to the
# path + a "csv, N columns, M rows" descriptor):
@plotter histogram of ages in @~/data/people.csv

# or run it yourself:
/plot ~/data/people.csv kind=histogram x=age bins=15
/plot ~/data/people.csv kind=line y=passengers       # row index on the x axis
/plot ~/data/people.csv x=height y=weight
/plot ~/data/people.csv x=pc1 y=pc2 z=pc3     # once your PCA tool lands
```

## Arguments

| field  | default  | notes                                             |
|--------|----------|---------------------------------------------------|
| `csv`  | —        | the data (or use `path`)                          |
| `path` | —        | host-resolved CSV file path                       |
| `kind` | `scatter`| `scatter` · `line` · `histogram` · `cdf`          |
| `x`    | index    | domain column (name/index); OMIT = row index      |
| `y`    | `1`      | value column (scatter/line)                       |
| `z`    | —        | column for a 3D scatter                           |
| `bins` | per-row  | histogram bins, 4..=200; omitted = one per row    |
| `pitch`/`yaw`/`distance` | 40/35/3.5 | 3D camera                    |
| `width`/`height` | 120×30 | letterbox default; cell cap 10 000       |

CSV parsing is deliberately small: comma-separated, `#` comments, no
quoted fields (yet); a first row with any non-numeric cell becomes the
header; rows with non-numeric cells in used columns are skipped and
counted.

## Build

```sh
cargo build -p plot-device-tool --release --target wasm32-wasip2
cp target/wasm32-wasip2/release/plot_device_tool.wasm runtime-tools/plot_device.wasm
```

Preview on the host:

```sh
cargo run -p plot-device-tool --example show_plot
```
