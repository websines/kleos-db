//! Reading order and layout blocks: line features for the order model, order
//! decoding, XY-cut, block clustering (`order_post.py`), and page Markdown
//! with tables (`layout_post.py`).

use super::text::{all_digits, display_len, is_alpha, is_digit, is_lower, is_space, is_upper};

pub type Bx = [f32; 4];

pub const N_FEATS: usize = 16;
pub const BLOCK_CLASSES: [&str; 10] = [
    "Text",
    "Section-Header",
    "Page-Header",
    "Page-Footer",
    "List-Group",
    "Table",
    "Caption",
    "Footnote",
    "Table-Of-Contents",
    "Form",
];
pub const GRID_LABELS: [&str; 3] = ["Table", "Form", "Table-Of-Contents"];
const SENTENCE_END: &str = ".!?:;。！？：；．";

pub fn class_id(label: &str) -> usize {
    BLOCK_CLASSES.iter().position(|&c| c == label).unwrap_or(0)
}

/// np.median.
pub fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

pub fn median_f64(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// [N, 16] features (row-major) for the order model.
pub fn line_features(boxes: &[Bx], wh: (f32, f32), texts: Option<&[String]>) -> Vec<f32> {
    let (w_, h_) = wh;
    let n = boxes.len();
    let mut out = vec![0f32; n * N_FEATS];
    if n == 0 {
        return out;
    }
    let ws: Vec<f32> = boxes.iter().map(|b| (b[2] - b[0]).max(1e-3)).collect();
    let hs: Vec<f32> = boxes.iter().map(|b| (b[3] - b[1]).max(1e-3)).collect();
    let med_h = median(&hs);
    for (i, b) in boxes.iter().enumerate() {
        let f = &mut out[i * N_FEATS..(i + 1) * N_FEATS];
        f[0] = b[0] / w_;
        f[1] = b[1] / h_;
        f[2] = b[2] / w_;
        f[3] = b[3] / h_;
        f[4] = (b[0] + b[2]) / (2.0 * w_);
        f[5] = (b[1] + b[3]) / (2.0 * h_);
        f[6] = ws[i] / w_;
        f[7] = (hs[i] / med_h).ln();
        f[8] = (ws[i] / hs[i]).ln() / 4.0;
        f[9] = med_h / h_ * 20.0;
        if let Some(t) = texts.and_then(|t| t.get(i)) {
            let t = t.trim_matches(is_space);
            if t.is_empty() {
                continue;
            }
            let chars: Vec<char> = t.chars().collect();
            f[10] = ((display_len(t) as f64).ln_1p() / 5.0) as f32;
            f[11] = chars.iter().filter(|&&c| is_digit(c)).count() as f32 / chars.len() as f32;
            let head: String = chars.iter().take(3).collect();
            let head = head.trim_end_matches(['.', ')']);
            f[12] = ("•-–▪◦*·→".contains(chars[0]) || (all_digits(head) && chars.len() > 3)) as u8 as f32;
            f[13] = t.ends_with('-') as u8 as f32;
            f[14] = SENTENCE_END.contains(*chars.last().unwrap()) as u8 as f32;
            f[15] = is_upper(chars[0]) as u8 as f32;
        }
    }
    out
}

fn argsort_stable(v: &[f64], desc: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| {
        let o = v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal);
        if desc {
            o.reverse()
        } else {
            o
        }
    });
    idx
}

/// Order from a pairwise logit matrix [n, n] (a[i, j] > 0: i before j).
pub fn decode_order(a: &[f32], n: usize, sweeps: usize) -> Vec<usize> {
    if n <= 1 {
        return (0..n).collect();
    }
    // log P(i before j) = -logaddexp(0, -a)
    let lp: Vec<f32> = a.iter().map(|&v| -((0f32).max(-v) + (-(0f32 - -v).abs()).exp().ln_1p())).collect();
    let mut key = vec![0f64; n];
    for i in 0..n {
        let mut rs = 0f32;
        let mut cs = 0f32;
        for j in 0..n {
            rs += lp[i * n + j];
            cs += lp[j * n + i];
        }
        key[i] = (-rs + cs) as f64;
    }
    let mut order = argsort_stable(&key, false);
    for _ in 0..sweeps {
        let mut improved = false;
        for x in order.clone() {
            let rest: Vec<usize> = order.iter().copied().filter(|&v| v != x).collect();
            let m = rest.len();
            let mut gains = vec![0f64; m + 1];
            // prefix sums of lp[rest, x] (x after these) and suffix sums of lp[x, rest]
            let mut pre = vec![0f64; m + 1];
            let mut acc = 0f32;
            for (k, &r) in rest.iter().enumerate() {
                acc += lp[r * n + x];
                pre[k + 1] = acc as f64;
            }
            let mut suf = vec![0f64; m + 1];
            let mut acc = 0f32;
            for k in (0..m).rev() {
                acc += lp[x * n + rest[k]];
                suf[k] = acc as f64;
            }
            for k in 0..=m {
                gains[k] = pre[k] + suf[k];
            }
            let mut best = 0;
            for k in 1..=m {
                if gains[k] > gains[best] {
                    best = k;
                }
            }
            let cur = order.iter().position(|&v| v == x).unwrap();
            if gains[best] > gains[cur] + 1e-6 {
                let mut o = rest.clone();
                o.insert(best, x);
                order = o;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    order
}

/// Rows top to bottom (centres within half a line height share a row), left to right.
pub fn order_top_down(boxes: &[Bx]) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }
    let hs: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
    let h = median(&hs);
    let mut idx: Vec<usize> = (0..boxes.len()).collect();
    let key = |i: usize| {
        let cy = (boxes[i][1] + boxes[i][3]) / 2.0;
        ((cy / (h * 0.5).max(1e-3)) as f64).round_ties_even() as i64
    };
    idx.sort_by(|&a, &b| key(a).cmp(&key(b)).then(boxes[a][0].partial_cmp(&boxes[b][0]).unwrap_or(std::cmp::Ordering::Equal)));
    idx
}

fn best_cut(b: &[Bx], axis: usize) -> (f32, i64, Vec<usize>) {
    let (lo, hi): (Vec<f32>, Vec<f32>) = if axis == 1 { (b.iter().map(|x| x[1]).collect(), b.iter().map(|x| x[3]).collect()) } else { (b.iter().map(|x| x[0]).collect(), b.iter().map(|x| x[2]).collect()) };
    let mut o: Vec<usize> = (0..b.len()).collect();
    o.sort_by(|&a, &c| lo[a].partial_cmp(&lo[c]).unwrap_or(std::cmp::Ordering::Equal));
    let mut end = hi[o[0]];
    let (mut best_gap, mut best_k) = (0f32, -1i64);
    for k in 1..o.len() {
        let gap = lo[o[k]] - end;
        if gap > best_gap {
            best_gap = gap;
            best_k = k as i64;
        }
        end = end.max(hi[o[k]]);
    }
    (best_gap, best_k, o)
}

/// Recursive XY-cut at the widest whitespace gap.
pub fn order_xycut(boxes: &[Bx]) -> Vec<usize> {
    fn rec(boxes: &[Bx], ids: Vec<usize>) -> Vec<usize> {
        if ids.len() <= 1 {
            return ids;
        }
        let b: Vec<Bx> = ids.iter().map(|&i| boxes[i]).collect();
        let (gy, ky, oy) = best_cut(&b, 1);
        let (gx, kx, ox) = best_cut(&b, 0);
        if ky < 0 && kx < 0 {
            return order_top_down(&b).into_iter().map(|i| ids[i]).collect();
        }
        let (k, o) = if gy >= gx { (ky, oy) } else { (kx, ox) };
        let k = k as usize;
        let mut out = rec(boxes, o[..k].iter().map(|&i| ids[i]).collect());
        out.extend(rec(boxes, o[k..].iter().map(|&i| ids[i]).collect()));
        out
    }
    rec(boxes, (0..boxes.len()).collect())
}

/// Blocks in reading order by recursive XY-cut (vertical pages).
pub fn xycut_blocks(boxes: &[Bx], min_gap: f32) -> Vec<Vec<usize>> {
    if boxes.is_empty() {
        return vec![];
    }
    let hs: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
    let h = median(&hs);
    fn rec(boxes: &[Bx], ids: Vec<usize>, h: f32, min_gap: f32) -> Vec<Vec<usize>> {
        let b: Vec<Bx> = ids.iter().map(|&i| boxes[i]).collect();
        if ids.len() > 1 {
            let (gy, ky, oy) = best_cut(&b, 1);
            let (gx, kx, ox) = best_cut(&b, 0);
            let (gap, k, o) = if gy >= gx { (gy, ky, oy) } else { (gx, kx, ox) };
            if k >= 0 && gap >= min_gap * h {
                let k = k as usize;
                let mut out = rec(boxes, o[..k].iter().map(|&i| ids[i]).collect(), h, min_gap);
                out.extend(rec(boxes, o[k..].iter().map(|&i| ids[i]).collect(), h, min_gap));
                return out;
            }
        }
        vec![order_xycut(&b).into_iter().map(|i| ids[i]).collect()]
    }
    rec(boxes, (0..boxes.len()).collect(), h, min_gap)
}

/// Average-linkage clustering of pairwise "same group" logits [n, n].
pub fn cluster(logits: &[f32], n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![];
    }
    let mut s: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
    for i in 0..n {
        s[i * n + i] = 0.0;
    }
    let mut size = vec![1f64; n];
    let mut alive = vec![true; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut m = s.clone();
    for i in 0..n {
        m[i * n + i] = f64::NEG_INFINITY;
    }
    let argmax_row = |m: &[f64], j: usize| -> usize {
        let r = &m[j * n..(j + 1) * n];
        let mut b = 0;
        for k in 1..n {
            if r[k] > r[b] {
                b = k;
            }
        }
        b
    };
    let mut best: Vec<usize> = (0..n).map(|j| argmax_row(&m, j)).collect();
    let mut best_v: Vec<f64> = (0..n).map(|j| m[j * n + best[j]]).collect();
    for _ in 0..n.saturating_sub(1) {
        let mut a = 0;
        for k in 1..n {
            if best_v[k] > best_v[a] {
                a = k;
            }
        }
        let b = best[a];
        if best_v[a] <= 0.0 {
            break;
        }
        let (a, b) = (a.min(b), a.max(b));
        for k in 0..n {
            s[a * n + k] += s[b * n + k];
        }
        for k in 0..n {
            s[k * n + a] += s[k * n + b];
        }
        s[a * n + a] = 0.0;
        size[a] += size[b];
        alive[b] = false;
        let mb = std::mem::take(&mut members[b]);
        members[a].extend(mb);
        let mut row: Vec<f64> = (0..n).map(|k| s[a * n + k] / (size[a] * size[k])).collect();
        for k in 0..n {
            if !alive[k] {
                row[k] = f64::NEG_INFINITY;
            }
        }
        row[a] = f64::NEG_INFINITY;
        for k in 0..n {
            m[a * n + k] = row[k];
            m[k * n + a] = row[k];
            m[b * n + k] = f64::NEG_INFINITY;
            m[k * n + b] = f64::NEG_INFINITY;
        }
        best_v[b] = f64::NEG_INFINITY;
        let mut ba = 0;
        for k in 1..n {
            if row[k] > row[ba] {
                ba = k;
            }
        }
        best[a] = ba;
        best_v[a] = row[ba];
        for j in 0..n {
            if alive[j] && j != a && (best[j] == a || best[j] == b) {
                best[j] = argmax_row(&m, j);
                best_v[j] = m[j * n + best[j]];
            }
        }
        for j in 0..n {
            if alive[j] && j != a && row[j] > best_v[j] {
                best[j] = a;
                best_v[j] = row[j];
            }
        }
    }
    (0..n)
        .filter(|&i| alive[i])
        .map(|i| {
            let mut v = members[i].clone();
            v.sort();
            v
        })
        .collect()
}

/// Reading order with the layout heads: blocks first, then lines inside blocks.
pub fn decode_blocks_order(a: &[f32], same: &[f32], boxes: &[Bx], cls: Option<&[f32]>, n: usize) -> Vec<usize> {
    if n <= 1 {
        return (0..n).collect();
    }
    let blocks = cluster(same, n);
    let grid: Vec<usize> = GRID_LABELS.iter().map(|c| class_id(c)).collect();
    let nc = BLOCK_CLASSES.len();
    let mut inner = vec![];
    for b in &blocks {
        let bx: Vec<Bx> = b.iter().map(|&i| boxes[i]).collect();
        let is_grid = cls.map(|c| {
            let mut sum = vec![0f32; nc];
            for &i in b {
                for k in 0..nc {
                    sum[k] += c[i * nc + k];
                }
            }
            grid.contains(&argmax_f32(&sum))
        });
        let local = if is_grid == Some(true) { order_top_down(&bx) } else { order_xycut(&bx) };
        inner.push(local.into_iter().map(|k| b[k]).collect::<Vec<_>>());
    }
    let m = blocks.len();
    // G a G^T with G the block membership scaled by 1 / block size
    let mut gab = vec![0f64; m * m];
    for (p, bp) in blocks.iter().enumerate() {
        for (q, bq) in blocks.iter().enumerate() {
            let mut s = 0f64;
            for &i in bp {
                for &j in bq {
                    s += a[i * n + j] as f64;
                }
            }
            gab[p * m + q] = s / (bp.len() * bq.len()) as f64;
        }
    }
    let mut ab = vec![0f32; m * m];
    for p in 0..m {
        for q in 0..m {
            let up_pq = if q > p { gab[p * m + q] } else { 0.0 };
            let up_qp = if p > q { gab[q * m + p] } else { 0.0 };
            ab[p * m + q] = (up_pq - up_qp) as f32;
        }
    }
    decode_order(&ab, m, 4).into_iter().flat_map(|k| inner[k].clone()).collect()
}

pub fn argmax_f32(v: &[f32]) -> usize {
    let mut b = 0;
    for k in 1..v.len() {
        if v[k] > v[b] {
            b = k;
        }
    }
    b
}

// ---------------------------------------------------------------- layout and Markdown

fn cjk(c: char) -> bool {
    let o = c as u32;
    (0x3000..=0x30FF).contains(&o) || (0x3400..=0x9FFF).contains(&o) || (0xF900..=0xFAFF).contains(&o) || (0xFF00..=0xFFEF).contains(&o) || (0x20000..=0x2FA1F).contains(&o)
}

/// Join the lines of one paragraph (hyphenated words, CJK without spaces).
pub fn join_lines<S: AsRef<str>>(texts: &[S]) -> String {
    let mut out = String::new();
    for t in texts {
        let t = t.as_ref().trim_matches(is_space);
        if t.is_empty() {
            continue;
        }
        let oc: Vec<char> = out.chars().collect();
        let t0 = t.chars().next().unwrap();
        if oc.len() > 1 && out.ends_with('-') && is_alpha(oc[oc.len() - 2]) && is_alpha(t0) && is_lower(t0) {
            out.pop();
            out.push_str(t);
        } else if !oc.is_empty() && cjk(*oc.last().unwrap()) && cjk(t0) {
            out.push_str(t);
        } else if out.is_empty() {
            out = t.to_string();
        } else {
            out.push(' ');
            out.push_str(t);
        }
    }
    out
}

/// Runs of the reading order with a positive same-block logit between neighbours.
pub fn split_blocks(order: &[usize], same: &[f32], n: usize) -> Vec<Vec<usize>> {
    let mut blocks: Vec<Vec<usize>> = vec![];
    for &i in order {
        if let Some(b) = blocks.last_mut() {
            if same[*b.last().unwrap() * n + i] > 0.0 {
                b.push(i);
                continue;
            }
        }
        blocks.push(vec![i]);
    }
    blocks
}

fn rows_of(boxes: &[Bx]) -> Vec<Vec<usize>> {
    let cy: Vec<f32> = boxes.iter().map(|b| (b[1] + b[3]) / 2.0).collect();
    let h = median(&boxes.iter().map(|b| b[3] - b[1]).collect::<Vec<_>>());
    let mut idx: Vec<usize> = (0..boxes.len()).collect();
    idx.sort_by(|&a, &b| cy[a].partial_cmp(&cy[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rows: Vec<Vec<usize>> = vec![];
    for i in idx {
        if let Some(r) = rows.last_mut() {
            let m = r.iter().map(|&k| cy[k] as f64).sum::<f64>() / r.len() as f64;
            if ((cy[i] as f64) - m).abs() < 0.5 * h as f64 {
                r.push(i);
                continue;
            }
        }
        rows.push(vec![i]);
    }
    for r in rows.iter_mut() {
        r.sort_by(|&a, &b| boxes[a][0].partial_cmp(&boxes[b][0]).unwrap_or(std::cmp::Ordering::Equal));
    }
    rows
}

fn merge_wrapped(rows: Vec<Vec<usize>>, boxes: &[Bx]) -> Vec<Vec<usize>> {
    if rows.len() < 3 {
        return rows;
    }
    let h = median(&boxes.iter().map(|b| b[3] - b[1]).collect::<Vec<_>>()) as f64;
    let gaps: Vec<f64> = rows
        .windows(2)
        .map(|w| {
            let nmin = w[1].iter().map(|&k| boxes[k][1]).fold(f32::INFINITY, f32::min);
            let cmax = w[0].iter().map(|&k| boxes[k][3]).fold(f32::NEG_INFINITY, f32::max);
            (nmin - cmax) as f64
        })
        .collect();
    let med = median_f64(&gaps);
    let mut out = vec![rows[0].clone()];
    let mut prev = rows[0].clone();
    for (g, r) in gaps.iter().zip(rows[1..].iter()) {
        if *g < 0.5 * med && *g < 0.6 * h && r.len() < prev.len() {
            out.last_mut().unwrap().extend(r.iter().copied());
        } else {
            out.push(r.clone());
        }
        prev = r.clone();
    }
    out
}

pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// An HTML table from cell lines (one text line per cell).
pub fn table_html(boxes: &[Bx], texts: &[String]) -> String {
    let rows = rows_of(boxes);
    let span_x = boxes.iter().map(|b| b[2]).fold(f32::NEG_INFINITY, f32::max) - boxes.iter().map(|b| b[0]).fold(f32::INFINITY, f32::min);
    let width = rows.iter().map(|r| r.len()).max().unwrap_or(1);
    if rows.len() > 2 && rows[0].len() == 1 && width > 1 && boxes[rows[0][0]][2] - boxes[rows[0][0]][0] > 0.6 * span_x {
        let t = rows[0][0];
        let keep: Vec<usize> = (0..texts.len()).filter(|&k| k != t).collect();
        let kb: Vec<Bx> = keep.iter().map(|&k| boxes[k]).collect();
        let kt: Vec<String> = keep.iter().map(|&k| texts[k].clone()).collect();
        return format!("{}\n\n{}", texts[t], table_html(&kb, &kt));
    }
    let full: Vec<&Vec<usize>> = rows.iter().filter(|r| r.len() == width).collect();
    let rows = merge_wrapped(rows.clone(), boxes);
    let bands: Vec<(f64, f64)> = (0..width)
        .map(|c| {
            let l: Vec<f64> = full.iter().map(|r| boxes[r[c]][0] as f64).collect();
            let rr: Vec<f64> = full.iter().map(|r| boxes[r[c]][2] as f64).collect();
            (median_f64(&l), median_f64(&rr))
        })
        .collect();
    let mut edges = vec![f64::NEG_INFINITY];
    for c in 0..width.saturating_sub(1) {
        edges.push((bands[c].1 + bands[c + 1].0) / 2.0);
    }
    edges.push(f64::INFINITY);
    // np.searchsorted(edges, x, side="right") - 1
    let band_of = |x: f64| -> i64 { edges.iter().filter(|&&e| e <= x).count() as i64 - 1 };
    let mut out = vec!["<table>".to_string()];
    for (ri, r) in rows.iter().enumerate() {
        let tag = if ri == 0 && rows.len() > 1 { "th" } else { "td" };
        let mut cells: std::collections::BTreeMap<usize, (usize, String)> = Default::default();
        let mut rs = r.clone();
        rs.sort_by(|&a, &b| (boxes[a][1], boxes[a][0]).partial_cmp(&(boxes[b][1], boxes[b][0])).unwrap_or(std::cmp::Ordering::Equal));
        for k in rs {
            let (x0, x1) = (boxes[k][0] as f64, boxes[k][2] as f64);
            let first = band_of(x0);
            let last = band_of(x1);
            let lo = first.max(0) as usize;
            let hi = (last.min(width as i64 - 1)).max(-1);
            let mut span: Vec<usize> = vec![];
            if hi >= lo as i64 {
                for c in lo..=(hi as usize) {
                    if x1.min(bands[c].1) - x0.max(bands[c].0) > 0.5 * (bands[c].1 - bands[c].0) {
                        span.push(c);
                    }
                }
            }
            if span.is_empty() {
                span.push((first.max(0) as usize).min(width - 1));
            }
            let c0 = span[0];
            let e = match cells.get(&c0) {
                None => (span.len(), texts[k].clone()),
                Some((n, t)) => ((*n).max(span.len()), format!("{t} {}", texts[k])),
            };
            cells.insert(c0, e);
        }
        let mut tds = String::new();
        let mut c = 0;
        while c < width {
            if let Some((n, text)) = cells.get(&c) {
                let attr = if *n > 1 { format!(" colspan=\"{n}\"") } else { String::new() };
                tds.push_str(&format!("<{tag}{attr}>{}</{tag}>", html_escape(text)));
                c += n;
            } else {
                tds.push_str(&format!("<{tag}></{tag}>"));
                c += 1;
            }
        }
        out.push(format!("<tr>{tds}</tr>"));
    }
    out.push("</table>".into());
    out.join("\n")
}

/// A layout block: label, box, newline-joined text.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Block {
    pub label: String,
    pub bbox: [f32; 4],
    pub text: String,
}

pub fn round1(v: f32) -> f32 {
    ((v as f64 * 10.0).round_ties_even() / 10.0) as f32
}

fn block_class(cls: &[f32], blk: &[usize]) -> usize {
    let nc = BLOCK_CLASSES.len();
    let mut sum = vec![0f32; nc];
    for &i in blk {
        for k in 0..nc {
            sum[k] += cls[i * nc + k];
        }
    }
    argmax_f32(&sum)
}

/// The layout blocks in reading order.
pub fn page_blocks(boxes: &[Bx], texts: &[String], order: &[usize], cls: &[f32], same: &[f32]) -> Vec<Block> {
    let n = boxes.len();
    split_blocks(order, same, n)
        .into_iter()
        .map(|blk| {
            let b = union_box(blk.iter().map(|&i| boxes[i]));
            Block {
                label: BLOCK_CLASSES[block_class(cls, &blk)].to_string(),
                bbox: [round1(b[0]), round1(b[1]), round1(b[2]), round1(b[3])],
                text: blk.iter().map(|&i| texts[i].as_str()).collect::<Vec<_>>().join("\n"),
            }
        })
        .collect()
}

pub fn union_box(it: impl Iterator<Item = Bx>) -> Bx {
    let mut b = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    for x in it {
        b[0] = b[0].min(x[0]);
        b[1] = b[1].min(x[1]);
        b[2] = b[2].max(x[2]);
        b[3] = b[3].max(x[3]);
    }
    b
}

fn whole_math(t: &str) -> bool {
    let t: Vec<char> = t.chars().collect();
    t.len() >= 3 && t[0] == '$' && *t.last().unwrap() == '$' && !t[1..t.len() - 1].contains(&'$')
}

fn item_start(t: &str) -> bool {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)").unwrap()).is_match(t)
}

/// Markdown for one page.  `cells` maps a line to its table cells when it holds several.
#[allow(clippy::too_many_arguments)]
pub fn page_markdown(
    boxes: &[Bx],
    texts: &[String],
    order: &[usize],
    cls: &[f32],
    same: &[f32],
    keep_furniture: bool,
    cells: &std::collections::HashMap<usize, Vec<(Bx, String)>>,
    pictures: Option<&[Bx]>,
) -> String {
    let n = boxes.len();
    let (header, footer, section, list, table) = (class_id("Page-Header"), class_id("Page-Footer"), class_id("Section-Header"), class_id("List-Group"), class_id("Table"));
    let mut parts: Vec<String> = vec![];
    let mut part_boxes: Vec<Bx> = vec![];
    for blk in split_blocks(order, same, n) {
        let c = block_class(cls, &blk);
        let tb: Vec<&String> = blk.iter().map(|&i| &texts[i]).collect();
        if (c == header || c == footer) && !keep_furniture {
            continue;
        }
        let part = if tb.iter().all(|t| whole_math(t.trim_matches(is_space))) {
            tb.iter()
                .map(|t| {
                    let t = t.trim_matches(is_space);
                    let inner: String = t.chars().skip(1).take(t.chars().count() - 2).collect();
                    format!("$${inner}$$")
                })
                .collect::<Vec<_>>()
                .join("\n")
        } else if c == section {
            format!("## {}", join_lines(&tb))
        } else if c == table && blk.len() > 1 {
            let (mut cb, mut ct) = (vec![], vec![]);
            for &i in &blk {
                match cells.get(&i) {
                    Some(v) if !v.is_empty() => {
                        for (b, t) in v {
                            cb.push(*b);
                            ct.push(t.clone());
                        }
                    }
                    _ => {
                        cb.push(boxes[i]);
                        ct.push(texts[i].clone());
                    }
                }
            }
            table_html(&cb, &ct)
        } else if c == list {
            let mut items: Vec<Vec<&String>> = vec![];
            for t in &tb {
                if items.is_empty() || item_start(t) {
                    items.push(vec![t]);
                } else {
                    items.last_mut().unwrap().push(t);
                }
            }
            items.iter().map(|it| join_lines(it)).collect::<Vec<_>>().join("\n")
        } else {
            join_lines(&tb)
        };
        parts.push(part);
        part_boxes.push(union_box(blk.iter().map(|&i| boxes[i])));
    }
    if let Some(pics) = pictures.filter(|p| !p.is_empty()) {
        let mut slots: std::collections::BTreeMap<usize, Vec<String>> = Default::default();
        for (k, p) in pics.iter().enumerate() {
            let cy = (p[1] + p[3]) / 2.0;
            let at = part_boxes.iter().position(|b| (b[1] + b[3]) / 2.0 > cy && p[2].min(b[2]) > p[0].max(b[0])).unwrap_or(parts.len());
            slots.entry(at).or_default().push(format!("![](picture-{k})"));
        }
        let mut merged = vec![];
        for j in 0..=parts.len() {
            if let Some(s) = slots.get(&j) {
                merged.extend(s.iter().cloned());
            }
            if j < parts.len() {
                merged.push(parts[j].clone());
            }
        }
        parts = merged;
    }
    parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_hyphenated_lines() {
        assert_eq!(join_lines(&["tumori-", "genesis is", "rare"]), "tumorigenesis is rare");
        assert_eq!(join_lines(&["港务局", "发布"]), "港务局发布");
    }

    #[test]
    fn order_follows_logits() {
        // a[i][j] > 0: i before j; want 2, 0, 1
        let n = 3;
        let mut a = vec![0f32; 9];
        let set = |a: &mut Vec<f32>, i: usize, j: usize| {
            a[i * n + j] = 5.0;
            a[j * n + i] = -5.0;
        };
        set(&mut a, 2, 0);
        set(&mut a, 0, 1);
        set(&mut a, 2, 1);
        assert_eq!(decode_order(&a, n, 4), vec![2, 0, 1]);
    }

    #[test]
    fn clusters_by_positive_mean() {
        let n = 4;
        let mut l = vec![-3f32; 16];
        for (i, j) in [(0, 1), (2, 3)] {
            l[i * n + j] = 4.0;
            l[j * n + i] = 4.0;
        }
        assert_eq!(cluster(&l, n), vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn xycut_two_columns() {
        let b = vec![[0., 0., 40., 10.], [60., 0., 100., 10.], [0., 12., 40., 22.], [60., 12., 100., 22.]];
        assert_eq!(order_xycut(&b), vec![0, 2, 1, 3]);
    }
}
