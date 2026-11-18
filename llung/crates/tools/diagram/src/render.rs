//! Layout and rendering: `Graph` → monospace box-drawing art.
//!
//! Strategy (deliberately simple, good at small diagrams):
//!   1. Flatten the item tree into layout units: nodes, plus "phantom"
//!      boxes that contain no nodes anywhere (nested empty boxes like
//!      Harness ⊃ Context ⊃ … — they get a slot like a node).
//!   2. Rank nodes by longest path down the edges; order inside a rank
//!      by box-contiguity then barycenter (reduces crossings).
//!   3. Grid-placed node boxes; edges route on per-edge "trunk" rows in
//!      the channel between ranks (no two trunks share a row).
//!   4. Boxes draw around the union of their members' rects, outermost
//!      first so inner borders win.

use std::collections::HashMap;

use crate::dsl::{BoxNode, Graph, Item};

/// A unit of layout: a node, or a box with no nodes anywhere inside.
#[derive(Clone, Debug)]
enum Unit {
    Node(String),
    /// Path of the (empty) box in the item tree.
    EmptyBox(Vec<String>),
}

#[derive(Clone, Copy, Debug, Default)]
struct Rect {
    x0: usize,
    y0: usize,
    x1: usize, // exclusive
    y1: usize, // exclusive
}

impl Rect {
    fn union(self, other: Rect) -> Rect {
        Rect {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }
    fn padded(self, dx: usize, dy: usize) -> Rect {
        Rect {
            x0: self.x0.saturating_sub(dx),
            y0: self.y0.saturating_sub(dy),
            x1: self.x1 + dx,
            y1: self.y1 + dy,
        }
    }
    fn width(self) -> usize {
        self.x1.saturating_sub(self.x0)
    }
    fn height(self) -> usize {
        self.y1.saturating_sub(self.y0)
    }
}

/// Collect layout units in declaration order (DFS).
fn units(g: &Graph) -> Vec<Unit> {
    fn walk<'a>(g: &'a Graph, items: &'a [Item], path: &mut Vec<String>, out: &mut Vec<Unit>) {
        for item in items {
            match item {
                Item::Node(id) => out.push(Unit::Node(id.clone())),
                Item::Box(b) => {
                    path.push(b.id.clone());
                    if g.box_members(path).is_empty() {
                        out.push(Unit::EmptyBox(path.clone()));
                    } else {
                        walk(g, &b.items, path, out);
                    }
                    path.pop();
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(g, &g.items, &mut Vec::new(), &mut out);
    // Nodes that exist only because an edge named them were never
    // declared as items — append them so they get placed and ranked.
    let mut seen = std::collections::HashSet::new();
    for u in &out {
        if let Unit::Node(id) = u {
            seen.insert(id.clone());
        }
    }
    for n in &g.nodes {
        if !seen.contains(&n.id) {
            out.push(Unit::Node(n.id.clone()));
        }
    }
    out
}

/// Path of enclosing boxes for a node ("" = top level), for contiguity.
fn box_path_of(g: &Graph, id: &str) -> String {
    for (path, b) in g.boxes_with_paths() {
        if g.box_members(&path).iter().any(|m| m == id) {
            return path.join("/");
        }
        let _ = b;
    }
    String::new()
}

/// Natural size of a node's box.
fn node_size(label: &str) -> (usize, usize) {
    (label.chars().count().max(4) + 4, 3)
}

/// Longest-path ranks; edges pointing "up" (cycles) stay best-effort.
fn assign_ranks(g: &Graph, order: &[Unit]) -> HashMap<String, usize> {
    let mut rank: HashMap<String, usize> = HashMap::new();
    for u in order {
        if let Unit::Node(id) = u {
            rank.entry(id.clone()).or_insert(0);
        }
    }
    let n = rank.len().max(1);
    for _ in 0..n {
        let mut changed = false;
        for e in &g.edges {
            let (Some(&r), Some(&t)) = (rank.get(&e.from), rank.get(&e.to)) else {
                continue;
            };
            if t < r + 1 {
                rank.insert(e.to.clone(), r + 1);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Box cohesion: pull each box's members toward the shallowest
    // band occupied by a box-mate (sources like a lone branch node
    // would otherwise float to rank 0 and stretch the cluster
    // rectangle across unrelated nodes). A pull is skipped when it
    // would violate an edge's downward constraint.
    for (path, _) in g.boxes_with_paths() {
        let members = g.box_members(&path);
        for m in &members {
            // Target: the shallowest band a box-MATE occupies.
            let Some(band) = members
                .iter()
                .filter(|o| *o != m)
                .filter_map(|o| rank.get(o).copied())
                .min()
            else {
                continue;
            };
            let Some(&r) = rank.get(m) else { continue };
            if r >= band {
                continue;
            }
            let ok = g.edges.iter().all(|e| {
                let keeps_down = if e.from == *m { band < *rank.get(&e.to).unwrap_or(&0) || band <= *rank.get(&e.to).unwrap_or(&0) } else { true };
                let keeps_up = if e.to == *m { *rank.get(&e.from).unwrap_or(&0) < band } else { true };
                keeps_down && keeps_up
            });
            if ok {
                rank.insert(m.clone(), band);
            }
        }
    }
    rank
}

/// Render the graph to box-drawing art no wider than `budget` columns.
pub fn render(g: &Graph, budget: usize) -> String {
    let us = units(g);

    // ── Ranks and within-rank order ──
    let ranks_map = assign_ranks(g, &us);
    let mut rank_of_unit = |u: &Unit| -> usize {
        match u {
            Unit::Node(id) => *ranks_map.get(id).unwrap_or(&0),
            Unit::EmptyBox(_) => 0,
        }
    };
    let max_rank = us.iter().map(&mut rank_of_unit).max().unwrap_or(0);
    let mut ranks: Vec<Vec<Unit>> = vec![Vec::new(); max_rank + 1];
    for u in &us {
        ranks[rank_of_unit(u)].push(u.clone());
    }

    // Box-contiguity key, then a few barycenter passes to tame crossings.
    let key = |u: &Unit| -> String {
        match u {
            Unit::Node(id) => box_path_of(g, id),
            Unit::EmptyBox(path) => path[..path.len().saturating_sub(1)].join("/"),
        }
    };
    for r in &mut ranks {
        let k = key;
        r.sort_by_key(|u| k(u));
    }
    // Barycenter passes: within each box group, nodes sort by mean
    // neighbour position; a few iterations tame crossings.
    for _ in 0..3 {
        // (rank, node id) -> column index.
        let mut pos: HashMap<(usize, String), usize> = HashMap::new();
        for (ri, r) in ranks.iter().enumerate() {
            for (ci, u) in r.iter().enumerate() {
                if let Unit::Node(id) = u {
                    pos.insert((ri, id.clone()), ci);
                }
            }
        }
        let bary = |id: &str, ri: usize, len: usize| -> f64 {
            let mut sum = 0.0;
            let mut cnt = 0;
            for e in &g.edges {
                let other = if e.from == id {
                    Some(&e.to)
                } else if e.to == id {
                    Some(&e.from)
                } else {
                    None
                };
                if let Some(o) = other {
                    for adj in [ri.wrapping_sub(1), ri + 1] {
                        if let Some(&p) = pos.get(&(adj, o.clone())) {
                            sum += p as f64;
                            cnt += 1;
                        }
                    }
                }
            }
            if cnt == 0 {
                len as f64 / 2.0
            } else {
                sum / cnt as f64
            }
        };
        for (ri, r) in ranks.iter_mut().enumerate() {
            let mut keyed: Vec<(String, f64, Unit)> = r
                .iter()
                .map(|u| {
                    let b = match u {
                        Unit::Node(id) => bary(id, ri, r.len()),
                        Unit::EmptyBox(_) => f64::MAX,
                    };
                    (key(u), b, u.clone())
                })
                .collect();
            keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
            *r = keyed.into_iter().map(|(_, _, u)| u).collect();
        }
    }

    // ── Column geometry ──
    // Width of each unit, before budget fitting.
    let unit_w = |u: &Unit| -> usize {
        match u {
            Unit::Node(id) => node_size(&g.node(id).map(|n| n.label.as_str()).unwrap_or(id)).0,
            Unit::EmptyBox(path) => empty_box_size(g, path).0,
        }
    };
    let mut widths: Vec<Vec<usize>> = ranks
        .iter()
        .map(|r| r.iter().map(unit_w).collect())
        .collect();

    let gap = 3usize;
    let total = |widths: &[Vec<usize>]| -> usize {
        widths
            .iter()
            .map(|r| r.iter().sum::<usize>() + gap * r.len().saturating_sub(1))
            .max()
            .unwrap_or(0)
    };
    // Fit the budget: shave the widest cells, truncating labels later.
    let mut w_total = total(&widths);
    while w_total > budget {
        let Some((ri, ci)) = widths
            .iter()
            .enumerate()
            .flat_map(|(ri, r)| r.iter().enumerate().map(move |(ci, &w)| (ri, ci, w)))
            .max_by_key(|&(_, _, w)| w)
            .map(|(ri, ci, _)| (ri, ci))
        else {
            break;
        };
        if widths[ri][ci] <= 6 {
            break;
        }
        widths[ri][ci] -= 1;
        w_total = total(&widths);
    }

    // ── Row geometry ──
    let unit_h = |u: &Unit| -> usize {
        match u {
            Unit::Node(_) => 3,
            Unit::EmptyBox(path) => empty_box_size(g, path).1,
        }
    };
    let mut row_of_rank: Vec<usize> = Vec::new();
    let mut rows: Vec<usize> = Vec::new();
    {
        let mut y = 0usize;
        for (ri, r) in ranks.iter().enumerate() {
            row_of_rank.push(y);
            let h = r.iter().map(unit_h).max().unwrap_or(3);
            rows.push(h);
            y += h;
            if ri < ranks.len() - 1 {
                let count = g
                    .edges
                    .iter()
                    .filter(|e| {
                        ranks_map.get(&e.from).copied().unwrap_or(0) == ri
                            && ranks_map.get(&e.to).copied().unwrap_or(0) > ri
                    })
                    .count();
                y += count.max(2); // channel rows between ranks
            }
        }
    }

    // ── Place units ──
    let mut canvas = Canvas::new();
    let mut node_rects: HashMap<String, Rect> = HashMap::new();
    let mut empty_rects: HashMap<String, Rect> = HashMap::new();
    for (ri, r) in ranks.iter().enumerate() {
        // Headroom (x≥2, y+1): a top-level box pads around its members,
        // so the canvas needs room outside the outermost unit positions.
        let mut x = 2usize;
        let y0 = row_of_rank[ri] + 1;
        for (ci, u) in r.iter().enumerate() {
            let w = widths[ri][ci];
            match u {
                Unit::Node(id) => {
                    let label = g.node(id).map(|n| n.label.as_str()).unwrap_or(id);
                    let rect = Rect { x0: x, y0, x1: x + w, y1: y0 + 3 };
                    node_rects.insert(id.clone(), rect);
                    let _ = label;
                }
                Unit::EmptyBox(path) => {
                    let h = unit_h(u);
                    empty_rects.insert(
                        path.join("/"),
                        Rect { x0: x, y0, x1: x + w, y1: y0 + h },
                    );
                }
            }
            x += w + gap;
        }
    }

    // ── Draw edges first (nodes and boxes overwrite them) ──
    for e in &g.edges {
        let (Some(src), Some(dst)) = (node_rects.get(&e.from), node_rects.get(&e.to)) else {
            continue;
        };
        let (sx, sy) = ((src.x0 + src.x1) / 2, src.y1 - 1); // bottom center
        let (tx, ty) = ((dst.x0 + dst.x1) / 2, dst.y0); // top center
        if ty > sy {
            // Downward: trunk row just below the source.
            let trunk = sy + 1;
            let v_end = ty - 1;
            for yy in sy + 1..=trunk.min(v_end) {
                canvas.set(sx, yy, '│');
            }
            let (x0, x1) = if sx <= tx { (sx, tx) } else { (tx, sx) };
            for xx in x0..=x1 {
                canvas.set(xx, trunk, '─');
            }
            canvas.set(sx, trunk, if sx <= tx { '└' } else { '┘' });
            canvas.set(tx, trunk, if sx <= tx { '┐' } else { '┌' });
            for yy in trunk + 1..v_end {
                canvas.set(tx, yy, '│');
            }
            canvas.set(tx, v_end, '▼');
            if let Some(label) = &e.label {
                let lx = x1 + 2;
                if lx + label.chars().count() <= budget {
                    canvas.text(lx, trunk, label);
                }
            }
        } else {
            // Upward (cycle): enter the target from below.
            let trunk = ty + 4; // one row below the target box
            for yy in ty + 3..sy {
                canvas.set(tx, yy, '│');
            }
            canvas.set(tx, ty + 3, '▲');
            let (x0, x1) = if sx <= tx { (sx, tx) } else { (tx, sx) };
            for xx in x0..=x1 {
                canvas.set(xx, trunk, '─');
            }
            canvas.set(tx, trunk, if sx <= tx { '┌' } else { '┐' });
            canvas.set(sx, trunk, if sx <= tx { '┘' } else { '└' });
        }
    }

    // ── Draw boxes (outermost first), then nodes ──
    let keyed: Vec<(Vec<String>, &BoxNode)> = g.boxes_with_paths();
    // Resolve rects: boxes whose members are placed, deepest first;
    // empty boxes are placed by their parent (or their unit rect).
    let mut resolved: HashMap<String, Rect> = HashMap::new();
    for _pass in 0..keyed.len() + 1 {
        let mut progress = false;
        for (path, b) in keyed.iter().rev() {
            let key_s = path.join("/");
            if resolved.contains_key(&key_s) {
                continue;
            }
            let members = g.box_members(path);
            let member_rects: Vec<Rect> = members
                .iter()
                .filter_map(|m| node_rects.get(m).copied())
                .collect();
            let child_rects: Vec<Rect> = b
                .items
                .iter()
                .filter_map(|it| match it {
                    Item::Box(cb) => {
                        resolved.get(&format!("{key_s}/{}", cb.id)).copied()
                    }
                    _ => None,
                })
                .collect();
            if members.is_empty() {
                // Empty box: placed by parent, or it's a top-level
                // phantom with a unit rect already.
                if let Some(r) = empty_rects.get(&key_s) {
                    resolved.insert(key_s.clone(), *r);
                    progress = true;
                }
                continue;
            }
            if member_rects.len() < members.len() || child_rects.len() < child_count(b) {
                continue; // members not placed yet
            }
            let mut rect = member_rects[0];
            for r in member_rects.iter().chain(child_rects.iter()).skip(1) {
                rect = rect.union(*r);
            }
            resolved.insert(key_s.clone(), rect.padded(2, 1));
            progress = true;
            let _ = b;
        }
        if !progress {
            break;
        }
    }
    let box_rects: HashMap<String, Rect>;
    // Place empty child boxes inside their parents (stacked), growing
    // the parent rect to include them.
    // Outermost first: a child's anchor is its parent's rect, so
    // parents must be resolved before their empty children.
    let mut pending: Vec<(String, Rect)> = Vec::new();
    for (path, b) in keyed.iter() {
        let key_s = path.join("/");
        if resolved.contains_key(&key_s) {
            continue;
        }
        let parent_key = path[..path.len() - 1].join("/");
        let (w, h) = empty_box_size(g, path);
        let anchor = if parent_key.is_empty() {
            empty_rects.get(&key_s).copied()
        } else {
            resolved.get(&parent_key).map(|r| Rect {
                x0: r.x0 + 2,
                y0: r.y0 + 2,
                x1: r.x0 + 2 + w,
                y1: r.y0 + 2 + h,
            })
        };
        if let Some(rect) = anchor {
            resolved.insert(key_s.clone(), rect);
            pending.push((parent_key, rect));
        }
        let _ = b;
    }
    for (parent, child) in pending {
        if let Some(r) = resolved.get(&parent).copied() {
            resolved.insert(parent, r.union(child).padded(2, 1));
        }
    }
    box_rects = resolved;

    // Draw outermost first so inner borders overwrite.
    let mut drawn: Vec<(usize, String, String)> = keyed
        .iter()
        .map(|(p, b)| (p.len(), p.join("/"), b.label.clone()))
        .collect();
    drawn.sort_by(|a, b| a.0.cmp(&b.0));
    for (_, key_s, label) in drawn {
        if let Some(&r) = box_rects.get(&key_s) {
            canvas.boxed(r, &label);
        }
    }

    // Nodes on top.
    for (id, rect) in &node_rects {
        let label = g.node(id).map(|n| n.label.as_str()).unwrap_or(id);
        let has_out = g.edges.iter().any(|e| &e.from == id);
        let has_in = g.edges.iter().any(|e| &e.to == id);
        canvas.node(*rect, label, rect.width(), has_in, has_out);
    }

    // Last-resort budget fit: hard-truncate any line still too wide.
    let out = canvas.to_string();
    out.lines()
        .map(|l| {
            if l.chars().count() > budget {
                l.chars().take(budget).collect::<String>()
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn child_count(b: &BoxNode) -> usize {
    b.items.iter().filter(|i| matches!(i, Item::Box(_))).count()
}

/// Measured size of an (empty) box: label + stacked children.
fn empty_box_size(g: &Graph, path: &[String]) -> (usize, usize) {
    let Some((_, b)) = g
        .boxes_with_paths()
        .into_iter()
        .find(|(p, _)| p.as_slice() == path)
    else {
        return (10, 3);
    };
    let mut w = b.label.chars().count() + 6;
    let mut h = 0usize;
    for item in &b.items {
        if let Item::Box(cb) = item {
            let (cw, chh) = empty_box_size(g, &[path, &[cb.id.clone()]].concat());
            w = w.max(cw + 4);
            h += chh + 1;
        }
    }
    if h == 0 {
        h = 1; // inner padding row
    }
    (w.max(10), h + 2)
}

/// A sparse char canvas that grows as needed.
#[derive(Default)]
struct Canvas {
    cells: HashMap<(usize, usize), char>,
    max_x: usize,
    max_y: usize,
}

impl Canvas {
    fn new() -> Self {
        Self::default()
    }
    fn set(&mut self, x: usize, y: usize, c: char) {
        if c == ' ' {
            return;
        }
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.cells.insert((x, y), c);
    }
    fn text(&mut self, x: usize, y: usize, s: &str) {
        for (i, c) in s.chars().enumerate() {
            self.set(x + i, y, c);
        }
    }
    /// A node box: top/mid/bottom rows, label centered/truncated.
    fn node(&mut self, r: Rect, label: &str, w: usize, has_in: bool, has_out: bool) {
        let w = w.max(6);
        let inner = w.saturating_sub(2);
        let label: String = label.chars().take(inner.saturating_sub(2)).collect();
        let pad = inner.saturating_sub(label.chars().count());
        let left = pad / 2;
        // top
        for xx in r.x0..r.x1 {
            self.set(xx, r.y0, '─');
        }
        self.set(r.x0, r.y0, '┌');
        self.set(r.x1 - 1, r.y0, '┐');
        // mid
        self.set(r.x0, r.y0 + 1, '│');
        self.set(r.x1 - 1, r.y0 + 1, '│');
        for (i, c) in label.chars().enumerate() {
            self.set(r.x0 + 2 + left + i, r.y0 + 1, c);
        }
        // bottom
        for xx in r.x0..r.x1 {
            self.set(xx, r.y1 - 1, '─');
        }
        self.set(r.x0, r.y1 - 1, '└');
        self.set(r.x1 - 1, r.y1 - 1, '┘');
        // edge attach points
        let cx = (r.x0 + r.x1) / 2;
        if has_in {
            self.set(cx, r.y0, '┬');
        }
        if has_out {
            self.set(cx, r.y1 - 1, '┴');
        }
    }
    /// A container box with its label on the top border.
    fn boxed(&mut self, r: Rect, label: &str) {
        if r.width() < 4 || r.height() < 3 {
            return;
        }
        for xx in r.x0..r.x1 {
            self.set(xx, r.y0, '─');
            self.set(xx, r.y1 - 1, '─');
        }
        for yy in r.y0..r.y1 {
            self.set(r.x0, yy, '│');
            self.set(r.x1 - 1, yy, '│');
        }
        self.set(r.x0, r.y0, '┌');
        self.set(r.x1 - 1, r.y0, '┐');
        self.set(r.x0, r.y1 - 1, '└');
        self.set(r.x1 - 1, r.y1 - 1, '┘');
        // label: ┌─ LABEL ───┐
        let label: String = label.chars().take(r.width().saturating_sub(6)).collect();
        if !label.is_empty() {
            self.text(r.x0 + 2, r.y0, &format!(" {label} "));
        }
    }
}

impl std::fmt::Display for Canvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..=self.max_y {
            let mut line = String::new();
            for x in 0..=self.max_x {
                line.push(self.cells.get(&(x, y)).copied().unwrap_or(' '));
            }
            let line = line.trim_end();
            if line.is_empty() {
                writeln!(f)?;
            } else {
                writeln!(f, "{line}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::parse;

    #[test]
    fn simple_flow_renders_boxes_and_arrows() {
        let g = parse("START: Kickoff\nEND: Done\nSTART -> END: go\n");
        let art = render(&g, 64);
        assert!(art.contains('▼'), "arrow drawn: {art}");
        assert!(art.contains("Kickoff"), "{art}");
        assert!(art.contains("Done"), "{art}");
        // Kickoff above Done (ranked flow).
        let (yk, yd) = (art.find("Kickoff"), art.find("Done"));
        assert!(yk.is_some() && yd.is_some() && yk < yd, "{art}");
        // No line exceeds the budget.
        assert!(art.lines().all(|l| l.chars().count() <= 64));
    }

    #[test]
    fn nested_empty_boxes() {
        let g = parse(
            "box HARNESS: Harness {\n  box CONTEXT: Context {\n    box PROMPT: Prompt {\n      box MODEL: Model {\n      }\n    }\n  }\n}\n",
        );
        let art = render(&g, 64);
        assert!(art.contains("Harness"), "{art}");
        assert!(art.contains("Model"), "{art}");
        // Every width within budget.
        assert!(art.lines().all(|l| l.chars().count() <= 64));
    }

    #[test]
    fn box_around_grouped_nodes() {
        let g = parse(
            "box INVESTIGATE: Investigate {\n  A: Sub Agent A\n  B: Sub Agent B\n}\nA -> B\nPLAN: Draft Plan\nB -> PLAN\n",
        );
        let art = render(&g, 72);
        assert!(art.contains("Investigate"), "{art}");
        assert!(art.contains("Sub Agent A"), "{art}");
        assert!(art.lines().all(|l| l.chars().count() <= 72));
    }

    #[test]
    fn budget_is_respected_when_wide() {
        let wide = (0..8)
            .map(|i| format!("N{i}: Node number {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let g = parse(&wide);
        let art = render(&g, 40);
        assert!(art.lines().all(|l| l.chars().count() <= 40), "{art}");
    }
}
