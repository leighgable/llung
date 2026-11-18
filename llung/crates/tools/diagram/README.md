# diagram-tool

llung wasm tool: flowcharts and nested-box diagrams as terminal
box-drawing art, directly in the chat. The model describes the graph in
a small line-based language; the tool lays it out (layered ranks,
crossing reduction, container rectangles) and returns monospace art that
renders in any terminal — pure text, no graphics protocol.

## Language

One statement per line:

```text
# comments start with #
box OUTER: Optional Label {   # a container; boxes nest freely
  A: First node               # node: `id: label`
  box INNER {                 # label defaults to the id
    B: Nested node
  }
}
A -> B: optional edge label   # edges; chains (A -> B -> C) work
GROUP -> C                    # an edge may name a box: from its last
C -> GROUP                    # inner node / to its first inner node
C                             # bare id = node labeled with itself
```

Lenient on purpose: edges auto-create undeclared nodes, a real
declaration replaces an edge-created stub, duplicate declarations keep
the first label, unclosed boxes close themselves, and cycles render
best-effort — a small model's typo degrades the diagram instead of
failing the turn.

## Arguments

| field   | default | notes                                   |
|---------|---------|-----------------------------------------|
| `graph` | —       | the diagram, in the language above      |
| `title` | —       | short panel subtitle                    |
| `width` | `64`    | max chars, clamped to `24..=110`        |

The art is plain text, so keep `width` within the chat panel (the chat
wraps overlong lines, which mangles box borders).

## Layout

- Nodes are ranked by longest path down the edges; within a rank they
  are ordered to keep box members adjacent and reduce crossings
  (barycenter passes).
- Box members with no incoming edge are pulled into their box-mates'
  band, so container rectangles stay tight.
- Edges route on their own trunk row between ranks with `▼`/`▲` arrows
  and optional labels; nodes and boxes are drawn over the lines.

## Build

Same procedure as the other tools (see
`crates/tools/landscape/README.md` for the nix-sysroot workaround):

```sh
cargo build -p diagram-tool --release --target wasm32-wasip2
cp target/wasm32-wasip2/release/diagram_tool.wasm runtime-tools/diagram.wasm
```

Preview on the host (no wasm needed):

```sh
cargo run -p diagram-tool --example show_diagram
```
