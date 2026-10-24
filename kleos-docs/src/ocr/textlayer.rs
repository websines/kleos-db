//! PDF text-layer runs and the checks that decide whether to trust them
//! (`textlayer.py`).

use crate::pdf::PageRef;

use super::cv::Gray;
use super::text::{self, category, is_rtl, is_space};

const MAX_INVISIBLE_FRAC: f64 = 0.05;
const MAX_BAD_CHAR_FRAC: f64 = 0.005;
const MAX_NO_INK_LINE_FRAC: f64 = 0.03;
const MAX_VISUAL_RTL_FRAC: f64 = 0.25;
const MIN_RTL_PAIRS: usize = 20;
pub const VISUAL_RTL: &str = "right-to-left text in visual order";
pub const BROKEN_FONT_MAP: &str = "broken font map";
const DARK: u8 = 160;

pub fn bad_char_count(c: char) -> usize {
    let cat = category(c);
    (c == '\u{FFFD}' || cat == "Co" || cat == "Cs" || cat == "Cn" || (cat == "Cc" && c != '\t')) as usize
}

/// One text-layer run.
#[derive(Clone, Debug)]
pub struct TextLine {
    pub bbox: [f64; 4],
    pub text: String,
    pub chars: Vec<(char, [f64; 4])>,
    pub quarter: i32,
    pub seq: Option<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub chars: usize,
    pub rotated: usize,
    pub bad: usize,
    pub quarters: [usize; 4],
    pub off_page: usize,
    pub steps: (usize, usize),
    pub turn: i32,
    pub frame: i32,
    pub vertical: bool,
}

#[derive(Clone, Debug)]
pub struct TextLayer {
    pub lines: Vec<TextLine>,
    pub size: (f64, f64),
    pub trusted: bool,
    pub reason: String,
    pub stats: Stats,
    pub rotated: Vec<TextLine>,
}

fn quarter(d: (f64, f64)) -> Option<i32> {
    let q = d.1.atan2(d.0) / (std::f64::consts::PI / 2.0);
    let k = q.round_ties_even();
    if (q - k).abs() < 0.01 {
        Some((k as i32).rem_euclid(4))
    } else {
        None
    }
}

fn turn_frame(px: f64, py: f64, w: f64, h: f64, turn: i32) -> (f64, f64) {
    match turn {
        1 => (h - py, px),
        2 => (w - px, h - py),
        3 => (py, w - px),
        _ => (px, py),
    }
}

pub fn vertical_layout(st: &Stats) -> bool {
    let (down, right) = st.steps;
    down >= 30 && down > 2 * right
}

pub fn page_turn(st: &Stats) -> i32 {
    let q = st.quarters;
    let mut k = 1;
    for i in 2..4 {
        if q[i] > q[k] {
            k = i;
        }
    }
    if q[k] as f64 > 0.5 * st.chars.max(1) as f64 {
        k as i32
    } else {
        0
    }
}

fn code_point(u: u32, u2: u32) -> (char, bool) {
    if (0xD800..=0xDBFF).contains(&u) && (0xDC00..=0xDFFF).contains(&u2) {
        return (char::from_u32(0x10000 + ((u - 0xD800) << 10) + (u2 - 0xDC00)).unwrap_or('\u{FFFD}'), true);
    }
    if (0xD800..=0xDFFF).contains(&u) {
        return ('\u{FFFD}', false);
    }
    (char::from_u32(u).unwrap_or('\u{FFFD}'), false)
}

/// Text runs that read left to right once the page is turned upright by `turn`.
pub fn extract_runs(page: &PageRef, turn: i32, frame: Option<i32>) -> (Vec<TextLine>, Stats) {
    let frame = frame.unwrap_or(turn);
    let raw = page.chars();
    let a = page.device_affine();
    // L = F A[:, :2] F with F = diag(1, -1)
    let l = [[a[0][0], -a[0][1]], [-a[1][0], a[1][1]]];
    let mut st = Stats::default();
    let (mut chars, mut boxes_raw, mut breaks, mut qs): (Vec<char>, Vec<[f64; 4]>, Vec<usize>, Vec<i32>) = (vec![], vec![], vec![], vec![]);
    let mut cur_q = 0;
    let mut low = false;
    let mut upright_cache: std::collections::HashMap<u32, Option<i32>> = Default::default();
    for i in 0..raw.len() {
        if low {
            low = false;
            continue;
        }
        let u = raw[i].unit;
        let u2 = if (0xD800..=0xDBFF).contains(&u) && i + 1 < raw.len() { raw[i + 1].unit } else { 0 };
        let (mut c, l2) = code_point(u, u2);
        low = l2;
        if c == '\r' || c == '\n' {
            breaks.push(chars.len());
            continue;
        }
        if c == '\x02' {
            c = '-';
        }
        if !is_space(c) {
            st.chars += 1;
            st.bad += bad_char_count(c);
            let ang = raw[i].angle;
            let q = *upright_cache.entry(ang.to_bits()).or_insert_with(|| {
                let (ca, sa) = ((ang as f64).cos(), (ang as f64).sin());
                quarter((l[0][0] * ca + l[0][1] * sa, l[1][0] * ca + l[1][1] * sa))
            });
            if let Some(q) = q {
                st.quarters[q as usize] += 1;
            }
            if q != Some(turn) {
                st.rotated += 1;
            }
            let Some(q) = q else {
                breaks.push(chars.len());
                continue;
            };
            cur_q = (q - turn).rem_euclid(4);
        }
        chars.push(c);
        qs.push(cur_q);
        boxes_raw.push(raw[i].bx.unwrap_or([0.0; 4]));
    }
    if chars.is_empty() {
        return (vec![], st);
    }
    let (w, h) = page.size();
    let n = chars.len();
    let mut boxes: Vec<[f64; 4]> = Vec::with_capacity(n);
    let mut pxs: Vec<[f64; 4]> = Vec::with_capacity(n);
    let mut pys: Vec<[f64; 4]> = Vec::with_capacity(n);
    for b in &boxes_raw {
        // raw is (l, b, r, t); corners x [l, r, l, r], y [b, b, t, t]
        let xs = [b[0], b[2], b[0], b[2]];
        let ys = [b[1], b[1], b[3], b[3]];
        let mut px = [0f64; 4];
        let mut py = [0f64; 4];
        for k in 0..4 {
            px[k] = a[0][0] * xs[k] + a[0][1] * ys[k] + a[0][2];
            py[k] = a[1][0] * xs[k] + a[1][1] * ys[k] + a[1][2];
        }
        let mn = |v: &[f64; 4]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = |v: &[f64; 4]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        boxes.push([mn(&px), mn(&py), mx(&px), mx(&py)]);
        pxs.push(px);
        pys.push(py);
    }
    let cx: Vec<f64> = boxes.iter().map(|b| (b[0] + b[2]) / 2.0).collect();
    let cy: Vec<f64> = boxes.iter().map(|b| (b[1] + b[3]) / 2.0).collect();
    let visible: Vec<bool> = (0..n).map(|i| cx[i] >= 0.0 && cx[i] <= w && cy[i] >= 0.0 && cy[i] <= h).collect();
    st.off_page = visible.iter().filter(|v| !**v).count();
    let mut ok = vec![true; n.saturating_sub(1)];
    for &b in &breaks {
        if b > 0 && b < n {
            ok[b - 1] = false;
        }
    }
    let (mut down, mut right) = (0, 0);
    for i in 0..n.saturating_sub(1) {
        let o = ok[i] && qs[i] == 0 && qs[i + 1] == 0 && !is_space(chars[i]) && !is_space(chars[i + 1]);
        let hh = (boxes[i][3] - boxes[i][1]).max(1e-3);
        let (dx, dy) = (cx[i + 1] - cx[i], cy[i + 1] - cy[i]);
        let o = o && dx.abs() <= 3.0 * hh && dy.abs() <= 3.0 * hh;
        if o && dy > 0.3 * hh && dx.abs() < 0.3 * hh {
            down += 1;
        }
        if o && dx.abs() > 0.3 * hh && dy.abs() < 0.3 * hh {
            right += 1;
        }
    }
    st.steps = (down, right);
    if frame != 0 {
        for i in 0..n {
            let mut tx = [0f64; 4];
            let mut ty = [0f64; 4];
            for k in 0..4 {
                let (a1, b1) = turn_frame(pxs[i][k], pys[i][k], w, h, frame);
                tx[k] = a1;
                ty[k] = b1;
            }
            boxes[i] = [
                tx.iter().cloned().fold(f64::INFINITY, f64::min),
                ty.iter().cloned().fold(f64::INFINITY, f64::min),
                tx.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ty.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ];
        }
    }
    let mut cuts: Vec<usize> = breaks.clone();
    for i in 1..n {
        if qs[i] != qs[i - 1] {
            cuts.push(i);
        }
    }
    cuts.push(n);
    cuts.sort();
    cuts.dedup();
    let mut runs = vec![];
    let mut start = 0;
    for end in cuts {
        let seg: Vec<(char, [f64; 4])> = (start..end).filter(|&i| visible[i]).map(|i| (chars[i], boxes[i])).collect();
        let mut q = if start < qs.len() { qs[start] } else { 0 };
        start = end;
        let ink: Vec<[f64; 4]> = seg.iter().filter(|(c, _)| !is_space(*c)).map(|(_, b)| *b).collect();
        if ink.is_empty() {
            continue;
        }
        let bx = [
            ink.iter().map(|v| v[0]).fold(f64::INFINITY, f64::min),
            ink.iter().map(|v| v[1]).fold(f64::INFINITY, f64::min),
            ink.iter().map(|v| v[2]).fold(f64::NEG_INFINITY, f64::max),
            ink.iter().map(|v| v[3]).fold(f64::NEG_INFINITY, f64::max),
        ];
        let joined: String = seg.iter().map(|(c, _)| *c).collect();
        let text = text::squash_spaces(&joined);
        if frame != turn && q == 0 && ink.len() > 1 {
            let c0 = ((ink[0][0] + ink[0][2]) / 2.0, (ink[0][1] + ink[0][3]) / 2.0);
            let c1 = ((ink[ink.len() - 1][0] + ink[ink.len() - 1][2]) / 2.0, (ink[ink.len() - 1][1] + ink[ink.len() - 1][3]) / 2.0);
            if (c1.1 - c0.1).abs() > 2.0 * (c1.0 - c0.0).abs() {
                q = 1;
            }
        }
        let seq = Some(runs.len() as i64);
        runs.push(TextLine { bbox: bx, text, chars: seg, quarter: q, seq });
    }
    (runs, st)
}

fn visual_rtl_frac(runs: &[TextLine]) -> (usize, f64) {
    let (mut pairs, mut right) = (0, 0);
    for r in runs {
        for w in r.chars.windows(2) {
            let ((a, ba), (b, bb)) = (w[0], w[1]);
            if is_rtl(a) && is_rtl(b) {
                pairs += 1;
                right += (bb[0] + bb[2] > ba[0] + ba[2]) as usize;
            }
        }
    }
    (pairs, right as f64 / pairs.max(1) as f64)
}

fn no_ink_frac(gray: &Gray, runs: &[TextLine], scale: f64) -> f64 {
    let mut empty = 0;
    for ln in runs {
        let [x0, y0, x1, y1] = ln.bbox;
        let r0 = ((y0 * scale) as i64).max(0) as usize;
        let r1 = (((y1 * scale) as i64) + 1).clamp(0, gray.h as i64) as usize;
        let c0 = ((x0 * scale) as i64).max(0) as usize;
        let c1 = (((x1 * scale) as i64) + 1).clamp(0, gray.w as i64) as usize;
        let mut any = false;
        'o: for y in r0..r1 {
            for x in c0..c1 {
                if gray.at(x, y) < DARK {
                    any = true;
                    break 'o;
                }
            }
        }
        if !any {
            empty += 1;
        }
    }
    empty as f64 / runs.len().max(1) as f64
}

/// Text-layer runs of a page and the verdict on them.  `gray` is a render at
/// `scale` px/pt, turned upright like the runs.
pub fn read_text_layer(page: &PageRef, gray: &Gray, scale: f64, turn: i32, frame: Option<i32>) -> TextLayer {
    let frame = frame.unwrap_or(turn);
    let (w, h) = page.size();
    let size = if frame % 2 == 1 { (h, w) } else { (w, h) };
    let inv = page.invisible_text_frac();
    if inv > MAX_INVISIBLE_FRAC {
        return TextLayer { lines: vec![], size, trusted: false, reason: format!("invisible text {:.0}% (OCR layer)", inv * 100.0), stats: Stats::default(), rotated: vec![] };
    }
    let (runs, mut st) = extract_runs(page, turn, Some(frame));
    st.turn = turn;
    st.frame = frame;
    let (rotated, runs): (Vec<TextLine>, Vec<TextLine>) = runs.into_iter().partition(|r| r.quarter != 0);
    let mk = |lines, trusted, reason: &str, st: Stats, rotated| TextLayer { lines, size, trusted, reason: reason.to_string(), stats: st, rotated };
    if runs.is_empty() {
        return mk(vec![], false, "no text layer", st, rotated);
    }
    if st.bad as f64 > MAX_BAD_CHAR_FRAC * st.chars.max(1) as f64 {
        return mk(runs, false, BROKEN_FONT_MAP, st, rotated);
    }
    let (pairs, visual) = if frame == turn { visual_rtl_frac(&runs) } else { (0, 0.0) };
    if pairs >= MIN_RTL_PAIRS && visual > MAX_VISUAL_RTL_FRAC {
        return mk(runs, false, VISUAL_RTL, st, rotated);
    }
    let empty = no_ink_frac(gray, &runs, scale);
    if empty > MAX_NO_INK_LINE_FRAC {
        let r = format!("{:.0}% of text runs have no ink", empty * 100.0);
        return mk(runs, false, &r, st, rotated);
    }
    mk(runs, true, "", st, rotated)
}
