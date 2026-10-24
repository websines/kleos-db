//! The page OCR pipeline (`pipeline.py`): orientation and skew, line
//! detection, CTC line recognition (script routing, LM beam search), line
//! splitting and tidying, reading order, layout blocks and Markdown.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::cv::{self, Border, Gray};
use super::lm::{beam_search, NgramLm};
use super::models::{Input, Model, ModelError, ShapedModel};
use super::order::{self, Block, Bx, BLOCK_CLASSES, GRID_LABELS};
use super::prep;
use super::text::{self, Vocab};

const ADAPTIVE_MIN_H: f32 = 9.0;
const ADAPTIVE_TARGET_H: f32 = 12.0;
const SPLIT_GAP: f64 = 3.0;
const CELL_GAP: f64 = 1.8;
const GUTTER_GAP: f64 = 1.6;
const GUTTER_EDGES: usize = 3;
const ADAPTIVE_MAX_SIDE: usize = 2048;
const ORIENT_SIDE: usize = 768;
const ORIENT_MIN: f32 = 0.95;
pub const VERTICAL: i32 = 4;
const MIN_CONF: f64 = 0.5;
const LM_GATE: f64 = 0.9;
const LM_ALPHA: f64 = 0.5;
const LM_BETA: f64 = 1.5;
const LM_BEAM: usize = 8;
const LM_CAND_FLOOR: f64 = 3e-3;
const ROUTE_SURE: f32 = 0.95;
const PICTURE_MIN: f32 = 0.5;
const PICTURE_MIN_AREA: f32 = 0.004;
const DESKEW_MAX: f64 = 45.0;
const DESKEW_MIN: f64 = 0.3;
const PREVIEW_SIDE: usize = 1600;

/// The Hugging Face repo and tag of the models this port was checked against.
pub const HF_REPO: &str = "datalab-to/indra-ocr";
pub const HF_REVISION: &str = "v1.1.0";
/// Files the pipeline needs (math models are not used).
pub const MODEL_FILES: &[&str] = &["det.onnx", "rec.onnx", "meta.json"];
pub const OPTIONAL_FILES: &[&str] = &[
    "order.onnx",
    "orient.onnx",
    "script.onnx",
    "rec_cjk.onnx",
    "rec_rtl.onnx",
    "rec_world.onnx",
    "lm.npz",
    "lm_cjk.npz",
    "lm_world.npz",
];

/// One recognized line.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Line {
    pub bbox: [f32; 4],
    pub text: String,
    pub conf: f32,
}

/// One page: lines in reading order, Markdown, layout blocks.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Page {
    pub lines: Vec<Line>,
    pub markdown: Option<String>,
    pub blocks: Vec<Block>,
    /// ocr | text | hybrid | error
    pub source: String,
    pub note: String,
    pub rotation: i32,
    pub skew: f64,
    pub vertical: bool,
    pub timings: HashMap<String, f64>,
    /// page size in pixels (as read)
    pub size: (usize, usize),
}

impl Page {
    pub fn text(&self) -> String {
        self.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n")
    }
}

/// A token of a line: a recognizer id, or a character (group recognizers and
/// PDF text layers), with its page x and, for text layers, the content index.
#[derive(Clone, Debug, PartialEq)]
pub enum TokV {
    Id(usize),
    Ch(String),
}

#[derive(Clone, Debug)]
pub struct Tok {
    pub v: TokV,
    pub x: f64,
    pub seq: Option<i64>,
}

/// (text, confidence, tokens) of one line.
pub type Rec = (String, f64, Vec<Tok>);

fn ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Parallel map over items on scoped threads (order kept).
pub fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    if threads <= 1 || items.len() < 2 {
        return items.iter().map(&f).collect();
    }
    let chunk = items.len().div_ceil(threads);
    std::thread::scope(|s| {
        let hs: Vec<_> = items.chunks(chunk).map(|c| s.spawn(|| c.iter().map(&f).collect::<Vec<R>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

fn ink_thumb(gray: &Gray, side: usize) -> Gray {
    let s = side as f64 / gray.h.max(gray.w) as f64;
    let w = ((gray.w as f64 * s).round_ties_even() as usize).max(1);
    let h = ((gray.h as f64 * s).round_ties_even() as usize).max(1);
    cv::resize_area(gray, w, h).inverted()
}

fn row_sharpness(ink: &Gray, a: f64) -> f64 {
    // the row sums of warp_affine(ink, rotation, constant 0 border), without
    // building the rotated image
    let m = cv::rotation_matrix(ink.w as f64 / 2.0, ink.h as f64 / 2.0, a);
    let p = cv::warp_affine_row_sums(ink, m, ink.w, ink.h);
    p.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum()
}

fn arange(a: f64, b: f64, step: f64) -> Vec<f64> {
    // np.arange(a, b + 1e-6, step)
    let n = (((b + 1e-6) - a) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| a + i as f64 * step).collect()
}

/// Degrees (counter-clockwise) that straighten the text rows of a page.
pub fn estimate_skew(gray: &Gray, side: usize, threads: usize) -> f64 {
    let fine = ink_thumb(gray, side);
    if fine.mean() < 0.003 * 255.0 {
        return 0.0;
    }
    let coarse = ink_thumb(gray, side / 2);
    let grid = arange(-DESKEW_MAX, DESKEW_MAX, 2.0);
    let scores = par_map(&grid, threads, |&a| row_sharpness(&coarse, a));
    let mut idx: Vec<usize> = (0..grid.len()).collect();
    idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal));
    // starts = {0.0, two best}: a set; then near angles in insertion order
    let mut starts: Vec<f64> = vec![0.0];
    for &i in idx.iter().rev().take(2) {
        if !starts.iter().any(|&s| s == grid[i]) {
            starts.push(grid[i]);
        }
    }
    let mut near: Vec<f64> = vec![];
    for &a0 in &starts {
        near.extend(arange(a0 - 2.0, a0 + 2.0, 0.5));
    }
    let mut uniq: Vec<f64> = vec![];
    for &a in &near {
        if !uniq.iter().any(|&u| u == a) {
            uniq.push(a);
        }
    }
    let vals = par_map(&uniq, threads, |&a| row_sharpness(&fine, a));
    let score_of = |a: f64| vals[uniq.iter().position(|&u| u == a).unwrap()];
    let mut a1 = near[0];
    for &a in &near {
        if score_of(a) > score_of(a1) {
            a1 = a;
        }
    }
    let last = arange(a1 - 0.5, a1 + 0.5, 0.1);
    let lv = par_map(&last, threads, |&a| row_sharpness(&fine, a));
    let mut best = 0;
    for k in 1..last.len() {
        if lv[k] > lv[best] {
            best = k;
        }
    }
    last[best]
}

pub fn preview(gray: &Gray, side: usize) -> Gray {
    let s = side as f64 / gray.h.max(gray.w) as f64;
    if s >= 1.0 {
        return gray.clone();
    }
    cv::resize_area(gray, ((gray.w as f64 * s).round_ties_even() as usize).max(1), ((gray.h as f64 * s).round_ties_even() as usize).max(1))
}

pub fn rotate_page(gray: &Gray, degrees: f64) -> Gray {
    let m = cv::rotation_matrix(gray.w as f64 / 2.0, gray.h as f64 / 2.0, degrees);
    cv::warp_affine(gray, m, gray.w, gray.h, Border::Replicate)
}

fn round1(v: f64) -> f32 {
    ((v * 10.0).round_ties_even() / 10.0) as f32
}

pub fn unrotate_box(b: &[f32; 4], degrees: f64, wh: (usize, usize)) -> [f32; 4] {
    let (w, h) = (wh.0 as f64, wh.1 as f64);
    let m = cv::rotation_matrix(w / 2.0, h / 2.0, -degrees);
    let xs = [b[0], b[2], b[0], b[2]];
    let ys = [b[1], b[1], b[3], b[3]];
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for k in 0..4 {
        let (x, y) = (xs[k] as f64, ys[k] as f64);
        let cx = m[0][0] * x + m[0][1] * y + m[0][2];
        let cy = m[1][0] * x + m[1][1] * y + m[1][2];
        x0 = x0.min(cx);
        x1 = x1.max(cx);
        y0 = y0.min(cy);
        y1 = y1.max(cy);
    }
    [round1(x0.clamp(0.0, w)), round1(y0.clamp(0.0, h)), round1(x1.clamp(0.0, w)), round1(y1.clamp(0.0, h))]
}

fn orient_input(gray: &Gray) -> Vec<f32> {
    let side = ORIENT_SIDE;
    let s = side as f64 / gray.h.max(gray.w) as f64;
    let small = cv::resize_area(gray, ((gray.w as f64 * s).round_ties_even() as usize).max(1), ((gray.h as f64 * s).round_ties_even() as usize).max(1));
    let mut out = vec![0f32; side * side];
    for y in 0..small.h.min(side) {
        for x in 0..small.w.min(side) {
            out[y * side + x] = 1.0 - small.at(x, y) as f32 / 255.0;
        }
    }
    out
}

struct Group {
    name: String,
    model: Model,
    vocab: Vocab,
}

struct Lm {
    lm: NgramLm,
    alpha: f64,
    beta: f64,
    gate: f64,
}

/// The loaded OCR models and settings.
pub struct Ocr {
    det: ShapedModel,
    rec: Model,
    order: Option<Model>,
    orient: Option<Model>,
    script: Option<Model>,
    groups: Vec<Group>,
    script_groups: Vec<String>,
    lms: HashMap<String, Lm>,
    pub vocab: Vocab,
    long_side: usize,
    height: usize,
    window: usize,
    pub adaptive: bool,
    pub furniture: bool,
    pub threads: usize,
    pub dir: PathBuf,
}

/// Recognizer selector: base (rec.onnx) or a group index.
#[derive(Clone, Copy, PartialEq)]
enum Which {
    Base,
    Group(usize),
}

impl Ocr {
    /// Load from a model folder (det.onnx, rec.onnx, meta.json and the optional files).
    pub fn load(dir: &Path, threads: usize) -> Result<Self, ModelError> {
        let need = |f: &str| -> Result<PathBuf, ModelError> {
            let p = dir.join(f);
            if p.exists() {
                Ok(p)
            } else {
                Err(ModelError::Load(p.display().to_string(), "missing".into()))
            }
        };
        let opt = |f: &str| -> Option<PathBuf> { Some(dir.join(f)).filter(|p| p.exists()) };
        let meta_s = std::fs::read_to_string(need("meta.json")?).map_err(|e| ModelError::Load("meta.json".into(), e.to_string()))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_s).map_err(|e| ModelError::Load("meta.json".into(), e.to_string()))?;
        let vocab = Vocab::from_json(meta["vocab"].as_str().unwrap_or("")).map_err(|e| ModelError::Load("meta.json".into(), e))?;
        let det = ShapedModel::load(&need("det.onnx")?)?;
        let rec = Model::load(&need("rec.onnx")?)?;
        let order = opt("order.onnx").map(|p| Model::load(&p)).transpose()?;
        let orient = opt("orient.onnx").map(|p| Model::load(&p)).transpose()?;
        let mut groups = vec![];
        if let Some(obj) = meta.as_object() {
            for k in obj.keys().filter(|k| k.starts_with("vocab_")) {
                let g = &k["vocab_".len()..];
                if let Some(p) = opt(&format!("rec_{g}.onnx")) {
                    let v = Vocab::from_json(obj[k].as_str().unwrap_or("")).map_err(|e| ModelError::Load(k.clone(), e))?;
                    groups.push(Group { name: g.to_string(), model: Model::load(&p)?, vocab: v });
                }
            }
        }
        let script = if groups.is_empty() { None } else { opt("script.onnx").map(|p| Model::load(&p)).transpose()? };
        let script_groups = meta
            .get("script_groups")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|s| s.as_str().unwrap_or("").to_string()).collect())
            .unwrap_or_else(|| vec!["base".into(), "cjk".into(), "rtl".into(), "other".into()]);
        let params = meta.get("lm_params").cloned().unwrap_or_default();
        let mut lms = HashMap::new();
        let names: Vec<String> = std::iter::once("base".to_string()).chain(groups.iter().map(|g| g.name.clone())).collect();
        for name in names {
            let f = if name == "base" { "lm.npz".to_string() } else { format!("lm_{name}.npz") };
            if let Some(p) = opt(&f) {
                let lm = NgramLm::load(&p).map_err(|e| ModelError::Load(f.clone(), e))?;
                let pp = params.get(&name);
                let g = |k: &str, d: f64| pp.and_then(|v| v.get(k)).and_then(|v| v.as_f64()).unwrap_or(d);
                lms.insert(name.clone(), Lm { lm, alpha: g("alpha", LM_ALPHA), beta: g("beta", LM_BETA), gate: g("gate", LM_GATE) });
            }
        }
        Ok(Ocr {
            det,
            rec,
            order,
            orient,
            script,
            groups,
            script_groups,
            lms,
            vocab,
            long_side: meta["det_long_side"].as_u64().unwrap_or(1024) as usize,
            height: meta["rec_height"].as_u64().unwrap_or(32) as usize,
            window: meta.get("rec_window").and_then(|v| v.as_u64()).unwrap_or(1) as usize,
            adaptive: true,
            furniture: false,
            threads: threads.max(1),
            dir: dir.to_path_buf(),
        })
    }

    /// Characters any recognizer knows (for the text-layer readability check).
    pub fn knows(&self, c: &str) -> bool {
        self.vocab.index.contains_key(c) || self.groups.iter().any(|g| g.vocab.index.contains_key(c))
    }

    fn detect_at(&self, gray: &Gray, long_side: usize) -> Result<(Vec<prep::DetBox>, f64, Option<(Vec<f32>, usize, usize)>), ModelError> {
        let (x, h, w, s) = prep::page_to_input(gray, long_side);
        let out = self.det.run(&[1, 1, h, w], x)?;
        let (shape, data) = &out[0];
        let (c, oh, ow) = (shape[1], shape[2], shape[3]);
        let n = oh * ow;
        let pic = if c > 3 { Some((data[3 * n..4 * n].to_vec(), oh, ow)) } else { None };
        Ok((prep::decode(&data[..3 * n], oh, ow), s, pic))
    }

    /// Text-line boxes (pixels) and the picture channel of the last pass.
    fn detect_inner(&self, gray: &Gray) -> Result<(Vec<prep::DetBox>, Option<(Vec<f32>, usize, usize, f64)>), ModelError> {
        let (mut boxes, mut s, mut pic) = self.detect_at(gray, self.long_side)?;
        if self.adaptive && boxes.len() >= 8 {
            let med_h = order::median(&boxes.iter().map(|b| b[3] - b[1]).collect::<Vec<_>>());
            if med_h < ADAPTIVE_MIN_H {
                let ls = ((ADAPTIVE_MAX_SIDE as f64).min(self.long_side as f64 * ADAPTIVE_TARGET_H as f64 / (med_h as f64).max(1.0)) as usize) / 32 * 32;
                if ls > self.long_side {
                    let r = self.detect_at(gray, ls)?;
                    boxes = r.0;
                    s = r.1;
                    pic = r.2;
                }
            }
        }
        let (w, h) = (gray.w as f32, gray.h as f32);
        for b in boxes.iter_mut() {
            for k in 0..4 {
                b[k] = (b[k] as f64 / s) as f32;
            }
            b[0] = b[0].clamp(0.0, w);
            b[2] = b[2].clamp(0.0, w);
            b[1] = b[1].clamp(0.0, h);
            b[3] = b[3].clamp(0.0, h);
        }
        Ok((boxes, pic.map(|(d, oh, ow)| (d, oh, ow, s))))
    }

    pub fn detect(&self, gray: &Gray) -> Result<Vec<prep::DetBox>, ModelError> {
        Ok(self.detect_inner(gray)?.0)
    }

    fn picture_boxes(logit: &[f32], oh: usize, ow: usize, s: f64, hw: (usize, usize)) -> Vec<Bx> {
        let thr = (PICTURE_MIN / (1.0 - PICTURE_MIN)).ln();
        let mask: Vec<bool> = logit.iter().map(|&v| v > thr).collect();
        let (_, stats) = cv::connected_components(&mask, ow, oh, true);
        let f = 2.0 / s;
        let min_area = PICTURE_MIN_AREA as f64 * hw.0 as f64 * hw.1 as f64;
        let mut out = vec![];
        for st in stats.iter().skip(1) {
            let b = [
                (st.left as f64 * f) as f32,
                (st.top as f64 * f) as f32,
                (((st.left + st.width) as f64 * f).min(hw.1 as f64)) as f32,
                (((st.top + st.height) as f64 * f).min(hw.0 as f64)) as f32,
            ];
            if ((b[2] - b[0]) as f64) * ((b[3] - b[1]) as f64) >= min_area {
                out.push(b);
            }
        }
        out
    }

    fn crops(&self, gray: &Gray, boxes: &[prep::DetBox], window: usize, dark: Option<&[bool]>) -> Vec<(Vec<f32>, usize)> {
        let idx: Vec<usize> = (0..boxes.len()).collect();
        par_map(&idx, self.threads, |&i| prep::crop_line(gray, &boxes[i][..4], self.height, window, dark.map(|d| d[i])))
    }

    fn model_of(&self, w: Which) -> &Model {
        match w {
            Which::Base => &self.rec,
            Which::Group(g) => &self.groups[g].model,
        }
    }

    /// (group per crop, sure) from script.onnx.
    fn line_groups(&self, crops: &[(Vec<f32>, usize)], batch: usize) -> Result<(Vec<String>, Vec<bool>), ModelError> {
        let mut out = vec!["base".to_string(); crops.len()];
        let mut sure = vec![true; crops.len()];
        let Some(script) = &self.script else { return Ok((out, sure)) };
        let widths: Vec<usize> = crops.iter().map(|c| c.1).collect();
        for idx in prep::width_batches(&widths, batch) {
            let refs: Vec<&(Vec<f32>, usize)> = idx.iter().map(|&j| &crops[j]).collect();
            let (arr, ws, wm) = prep::pad_batch(&refs, self.height);
            let r = script.run(vec![Input::F32(vec![idx.len(), 1, self.height, wm], arr), Input::I64(vec![idx.len()], ws)])?;
            let p = r[0].f32();
            let shape = r[0].shape().to_vec();
            for (k, &j) in idx.iter().enumerate() {
                let (g, pk) = if shape.len() == 1 {
                    let v = p[k];
                    (if v > 0.5 { "cjk".to_string() } else { "base".to_string() }, v.max(1.0 - v))
                } else {
                    let nc = shape[1];
                    let row = &p[k * nc..(k + 1) * nc];
                    let a = order::argmax_f32(row);
                    (self.script_groups.get(a).cloned().unwrap_or_else(|| "base".into()), row[a])
                };
                out[j] = if self.groups.iter().any(|gr| gr.name == g) { g } else { "base".into() };
                sure[j] = pk > ROUTE_SURE;
            }
        }
        Ok((out, sure))
    }

    /// line_groups for boxes on the page.
    pub fn box_groups(&self, gray: &Gray, boxes: &[prep::DetBox]) -> Result<(Vec<String>, Vec<bool>), ModelError> {
        if self.script.is_none() || boxes.is_empty() {
            return Ok((vec!["base".into(); boxes.len()], vec![false; boxes.len()]));
        }
        let dark: Option<Vec<bool>> = if self.window > 1 { Some(par_map(boxes, self.threads, |b| prep::is_dark_line(gray, &b[..4]))) } else { None };
        let crops = self.crops(gray, boxes, 1, dark.as_deref());
        self.line_groups(&crops, 64)
    }

    /// (text, confidence, tokens) per line box.
    pub fn recognize(&self, gray: &Gray, boxes: &[prep::DetBox], vertical: bool) -> Result<Vec<Rec>, ModelError> {
        let batch = 64;
        let n = boxes.len();
        let dark: Option<Vec<bool>> = if self.window > 1 { Some(par_map(boxes, self.threads, |b| prep::is_dark_line(gray, &b[..4]))) } else { None };
        let crops = self.crops(gray, boxes, 1, dark.as_deref());
        let tall = if self.window > 1 { Some(self.crops(gray, boxes, self.window, dark.as_deref())) } else { None };
        let mut out: Vec<Rec> = vec![(String::new(), 0.0, vec![]); n];
        let mut plan: Vec<(Which, Vec<usize>, bool)> = vec![(Which::Base, (0..n).collect(), false)];
        let cjk = self.groups.iter().position(|g| g.name == "cjk");
        if vertical && cjk.is_some() {
            plan = vec![(Which::Group(cjk.unwrap()), (0..n).collect(), true)];
        } else if self.script.is_some() && n > 0 {
            let (names, sure) = self.line_groups(&crops, batch)?;
            plan = vec![(Which::Base, (0..n).filter(|&j| names[j] == "base" || !sure[j]).collect(), false)];
            for (gi, g) in self.groups.iter().enumerate() {
                plan.push((Which::Group(gi), (0..n).filter(|&j| names[j] == g.name).collect(), true));
            }
        }
        let mut best_conf = vec![-1f64; n];
        for (which, sel, as_chars) in plan {
            let widths: Vec<usize> = sel.iter().map(|&j| crops[j].1).collect();
            let model = self.model_of(which);
            let vocab = match which {
                Which::Base => &self.vocab,
                Which::Group(g) => &self.groups[g].vocab,
            };
            let lm_name = match which {
                Which::Base => "base".to_string(),
                Which::Group(g) => self.groups[g].name.clone(),
            };
            let lm = self.lms.get(&lm_name);
            for part in prep::width_batches(&widths, batch) {
                let idx: Vec<usize> = part.iter().map(|&i| sel[i]).collect();
                let src = if which == Which::Base { tall.as_ref().unwrap_or(&crops) } else { &crops };
                let hgt = if which == Which::Base && tall.is_some() { self.height * self.window } else { self.height };
                let refs: Vec<&(Vec<f32>, usize)> = idx.iter().map(|&j| &src[j]).collect();
                let (arr, ws, wm) = prep::pad_batch(&refs, hgt);
                let b = idx.len();
                let res = model.run(vec![Input::F32(vec![b, 1, hgt, wm], arr), Input::I64(vec![b], ws)])?;
                let best = res[0].i64();
                let tt = res[0].shape()[1];
                let pmax = res[1].f32();
                let lengths = res[2].i64();
                let stride = wm as f64 / (tt.max(1)) as f64;
                let (blank, topv, topi) = if lm.is_some() && res.len() >= 6 { (Some(res[3].f32()), Some(res[4].f32()), Some(res[5].i64())) } else { (None, None, None) };
                let kk = if res.len() >= 6 { res[4].shape()[2] } else { 0 };
                for (k, &j) in idx.iter().enumerate() {
                    let nlen = lengths[k] as usize;
                    let seq = &best[k * tt..k * tt + nlen];
                    let pm = &pmax[k * tt..k * tt + nlen];
                    let keep: Vec<bool> = (0..nlen).map(|t| seq[t] != 0 && (t == 0 || seq[t] != seq[t - 1])).collect();
                    let (x0, _y0, x1, _y1) = (boxes[j][0] as f64, boxes[j][1] as f64, boxes[j][2] as f64, boxes[j][3] as f64);
                    let h = (boxes[j][3] as f64 - boxes[j][1] as f64).max(1.0);
                    let ix0 = (x0 - prep::PAD_X as f64 * h).floor();
                    let cw = (x1 + prep::PAD_X as f64 * h).ceil() - ix0;
                    let f = cw / crops[j].1 as f64;
                    let mut ts: Vec<usize> = (0..nlen).filter(|&t| keep[t]).collect();
                    let mut ids: Vec<usize> = ts.iter().map(|&t| seq[t] as usize).collect();
                    let kept: Vec<f64> = (0..nlen).filter(|&t| keep[t]).map(|t| pm[t] as f64).collect();
                    // np.mean on float32 values (pairwise sum in f32)
                    let conf = if kept.is_empty() { 0.0 } else { (kept.iter().map(|&v| v as f32).sum::<f32>() / kept.len() as f32) as f64 };
                    if let (Some(l), Some(bl), Some(tv), Some(ti)) = (lm, &blank, &topv, &topi) {
                        if MIN_CONF <= conf && conf < l.gate {
                            let bk = &bl[k * tt..k * tt + nlen];
                            let tvk: Vec<Vec<f32>> = (0..nlen).map(|t| tv[(k * tt + t) * kk..(k * tt + t + 1) * kk].to_vec()).collect();
                            let tik: Vec<Vec<i64>> = (0..nlen).map(|t| ti[(k * tt + t) * kk..(k * tt + t + 1) * kk].to_vec()).collect();
                            let (bi, steps) = beam_search(bk, &tvk, &tik, &l.lm, l.alpha, l.beta, LM_BEAM, 0.999, LM_CAND_FLOOR);
                            ids = bi;
                            ts = steps;
                        }
                    }
                    let xs: Vec<f64> = ts.iter().map(|&t| ix0 + (t as f64 + 0.5) * stride * f).collect();
                    let toks: Vec<Tok> = ids
                        .iter()
                        .zip(xs.iter())
                        .map(|(&i, &x)| Tok { v: if as_chars { TokV::Ch(vocab.chars[i - 1].clone()) } else { TokV::Id(i) }, x, seq: None })
                        .collect();
                    if conf <= best_conf[j] {
                        continue;
                    }
                    best_conf[j] = conf;
                    out[j] = if conf >= MIN_CONF { (vocab.decode(&ids), conf, toks) } else { (String::new(), conf, vec![]) };
                }
            }
        }
        Ok(out)
    }

    fn is_space(&self, t: &TokV) -> bool {
        match t {
            TokV::Ch(c) => c == " ",
            TokV::Id(i) => Some(*i) == self.vocab.space(),
        }
    }

    /// Text of tokens.
    pub fn decode_toks(&self, toks: &[Tok], page_rtl: bool) -> String {
        if let Some(Tok { v: TokV::Ch(_), seq, .. }) = toks.first() {
            let joined: String = toks
                .iter()
                .map(|t| match &t.v {
                    TokV::Ch(c) => c.as_str(),
                    TokV::Id(_) => "",
                })
                .collect();
            let mut text = text::squash_spaces(&joined);
            if seq.is_none() {
                text = text::nfc(&text);
                if text::has_rtl(&text) {
                    text = text::flip(&text, if page_rtl { Some(true) } else { None });
                }
            }
            return text;
        }
        let ids: Vec<usize> = toks
            .iter()
            .filter_map(|t| match t.v {
                TokV::Id(i) => Some(i),
                _ => None,
            })
            .collect();
        self.vocab.decode(&ids)
    }

    fn gutter_cuts(&self, boxes: &[Bx], rec: &[Rec]) -> Vec<Vec<f64>> {
        let lefts: Vec<f32> = boxes.iter().map(|b| b[0]).collect();
        let rights: Vec<f32> = boxes.iter().map(|b| b[2]).collect();
        let mut cuts = vec![];
        for (i, (b, r)) in boxes.iter().zip(rec.iter()).enumerate() {
            let h = ((b[3] - b[1]) as f64).max(1.0);
            let xs: Vec<f64> = r.2.iter().filter(|t| !self.is_space(&t.v)).map(|t| t.x).collect();
            let mut mine = vec![];
            for w in xs.windows(2) {
                let (a, c) = (w[0], w[1]);
                if c - a <= GUTTER_GAP * h {
                    continue;
                }
                let (gl, gr) = (a + 0.35 * h, c - 0.35 * h);
                let near = (0..boxes.len()).filter(|&k| k != i && (((rights[k] as f64) - gl).abs() < 0.5 * h || ((lefts[k] as f64) - gr).abs() < 0.5 * h)).count();
                if near >= GUTTER_EDGES {
                    mine.push((a + c) / 2.0);
                }
            }
            cuts.push(mine);
        }
        cuts
    }

    /// Cut a line where neighbouring characters are more than `gap` line heights apart.
    pub fn split_line(&self, b: &Bx, toks: &[Tok], gap: f64, at: &[f64], page_rtl: bool) -> Vec<(Bx, String, Vec<Tok>)> {
        let h = ((b[3] - b[1]) as f64).max(1.0);
        let mut segs: Vec<Vec<Tok>> = vec![vec![]];
        let mut prev: Option<f64> = None;
        let mut at = at.to_vec();
        at.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for tk in toks {
            let x = tk.x;
            if !self.is_space(&tk.v) {
                if let Some(p) = prev {
                    if x - p > gap * h || at.iter().any(|&c| p < c && c < x) {
                        segs.push(vec![]);
                    }
                }
                prev = Some(x);
            }
            segs.last_mut().unwrap().push(tk.clone());
        }
        if segs.len() == 1 {
            return vec![(*b, self.decode_toks(toks, page_rtl), toks.to_vec())];
        }
        let mut out = vec![];
        for sg in segs {
            let t = self.decode_toks(&sg, page_rtl);
            let t = t.trim_matches(text::is_space).to_string();
            let xs: Vec<f64> = sg.iter().filter(|t| !self.is_space(&t.v)).map(|t| t.x).collect();
            if !t.is_empty() && !xs.is_empty() {
                let mn = xs.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let bx = [(b[0] as f64).max(mn - 0.35 * h) as f32, b[1], (b[2] as f64).min(mx + 0.35 * h) as f32, b[3]];
                out.push((bx, t, sg));
            }
        }
        out
    }

    /// Reading order, layout (cls [n, C], same [n, n]) when the order model has layout heads.
    fn reading_order(&self, boxes: &[Bx], texts: &[String], wh: (usize, usize)) -> Result<(Vec<usize>, Option<(Vec<f32>, Vec<f32>)>), ModelError> {
        let n = boxes.len();
        let Some(om) = self.order.as_ref().filter(|_| n >= 2) else {
            return Ok((if n > 0 { order::order_xycut(boxes) } else { vec![] }, None));
        };
        let f = order::line_features(boxes, (wh.0 as f32, wh.1 as f32), Some(texts));
        let outs = om.run(vec![Input::F32(vec![1, n, order::N_FEATS], f), Input::Bool(vec![1, n], vec![true; n])])?;
        let a = outs[0].f32();
        if outs.len() >= 3 {
            let cls = outs[1].f32();
            let same = outs[2].f32();
            let o = order::decode_blocks_order(&a, &same, boxes, Some(&cls), n);
            return Ok((o, Some((cls, same))));
        }
        Ok((order::decode_order(&a, n, 4), None))
    }

    fn orientation(&self, gray: &Gray) -> Result<i32, ModelError> {
        let Some(om) = &self.orient else { return Ok(0) };
        let r = om.run(vec![Input::F32(vec![1, 1, ORIENT_SIDE, ORIENT_SIDE], orient_input(gray))])?;
        let p = r[0].f32();
        let k = order::argmax_f32(&p);
        Ok(if p[k] > ORIENT_MIN { k as i32 } else { 0 })
    }

    fn turn(&self, gray: &Gray, k: i32, a: f64, small: Option<&Gray>) -> (Gray, i32, f64) {
        let q = if k == VERTICAL { 1 } else { -k };
        let mut g = if k == VERTICAL {
            cv::rot90(gray, 1)
        } else if k != 0 {
            cv::rot90(gray, -k)
        } else {
            gray.clone()
        };
        let mut a = a;
        if [1, 3, VERTICAL].contains(&k) {
            a = match small {
                Some(s) => estimate_skew(&cv::rot90(s, q), 800, self.threads),
                None => estimate_skew(&g, 800, self.threads),
            };
        }
        if a.abs() >= DESKEW_MIN {
            g = rotate_page(&g, a);
        } else {
            a = 0.0;
        }
        (g, k, a)
    }

    /// OCR of a page image.  `turn = false`: the page is known to be upright.
    pub fn read(&self, gray: &Gray, turn: bool) -> Result<Page, ModelError> {
        let t0 = std::time::Instant::now();
        let small = preview(gray, PREVIEW_SIDE);
        let a = estimate_skew(&small, 800, self.threads);
        let k = if turn {
            let s = if a.abs() >= DESKEW_MIN { rotate_page(&small, a) } else { small.clone() };
            self.orientation(&s)?
        } else {
            0
        };
        let orient_ms = ms(t0);
        let mut page = self.read_turned(gray, k, a, Some(&small))?;
        *page.timings.entry("orient_ms".into()).or_insert(0.0) += orient_ms;
        Ok(page)
    }

    fn read_turned(&self, gray: &Gray, k: i32, a: f64, small: Option<&Gray>) -> Result<Page, ModelError> {
        let mut t: HashMap<String, f64> = HashMap::new();
        let t0 = std::time::Instant::now();
        let (g, k, skew) = self.turn(gray, k, a, small);
        t.insert("orient_ms".into(), ms(t0));
        let t0 = std::time::Instant::now();
        let (boxes, pic) = self.detect_inner(&g)?;
        let pictures = match pic {
            Some((d, oh, ow, s)) => Self::picture_boxes(&d, oh, ow, s, (g.h, g.w)),
            None => vec![],
        };
        t.insert("det_ms".into(), ms(t0));
        let t0 = std::time::Instant::now();
        let vertical = k == VERTICAL;
        let rec = self.recognize(&g, &boxes, vertical)?;
        t.insert("rec_ms".into(), ms(t0));
        let bx: Vec<Bx> = boxes.iter().map(|b| [b[0], b[1], b[2], b[3]]).collect();
        let mut page = self.assemble((g.w, g.h), bx, rec, t, vertical, if vertical { None } else { Some(&pictures) })?;
        page.rotation = if vertical { 0 } else { 90 * k };
        page.skew = (skew * 100.0).round_ties_even() / 100.0;
        page.vertical = vertical;
        page.source = "ocr".into();
        page.size = (gray.w, gray.h);
        if skew != 0.0 {
            let wh = (g.w, g.h);
            for l in page.lines.iter_mut() {
                l.bbox = unrotate_box(&l.bbox, skew, wh);
            }
            for b in page.blocks.iter_mut() {
                b.bbox = unrotate_box(&b.bbox, skew, wh);
            }
        }
        if vertical {
            let w = g.h as f32;
            let back = |b: [f32; 4]| [w - b[3], b[0], w - b[1], b[2]];
            for l in page.lines.iter_mut() {
                l.bbox = back(l.bbox);
            }
            for b in page.blocks.iter_mut() {
                b.bbox = back(b.bbox);
            }
        }
        Ok(page)
    }

    /// Lines -> split at wide gaps -> reading order -> markdown.
    pub fn assemble(&self, wh: (usize, usize), boxes: Vec<Bx>, rec: Vec<Rec>, mut t: HashMap<String, f64>, vertical: bool, pictures: Option<&[Bx]>) -> Result<Page, ModelError> {
        let t0 = std::time::Instant::now();
        let letters: Vec<char> = rec.iter().flat_map(|r| r.0.chars()).filter(|&c| text::is_alpha(c)).collect();
        let n_rtl = letters.iter().filter(|&&c| text::is_rtl(c)).count();
        let page_rtl = n_rtl > letters.len() - n_rtl;
        let width = wh.0 as f32;
        let geo = |b: &Bx| -> Bx {
            if page_rtl {
                [width - b[2], b[1], width - b[0], b[3]]
            } else {
                *b
            }
        };
        let cuts = if boxes.is_empty() { vec![] } else { self.gutter_cuts(&boxes, &rec) };
        let (mut nb, mut nrec) = (vec![], vec![]);
        for ((b, r), at) in boxes.iter().zip(rec.iter()).zip(cuts.iter()) {
            for (bx, tx, tk) in self.split_line(b, &r.2, SPLIT_GAP, at, page_rtl) {
                nb.push(bx);
                nrec.push((tx, r.1, tk));
            }
        }
        let (mut boxes, mut rec) = self.tidy_lines(nb, nrec, page_rtl);
        let mut texts: Vec<String> = rec.iter().map(|r| r.0.clone()).collect();
        let gb: Vec<Bx> = boxes.iter().map(geo).collect();
        let (mut ord, mut layout) = self.reading_order(&gb, &texts, wh)?;
        let nc = BLOCK_CLASSES.len();
        if vertical && layout.is_some() {
            let blocks = order::xycut_blocks(&boxes, 1.0);
            ord = blocks.iter().flatten().copied().collect();
            let n = boxes.len();
            let mut same = vec![-1f32; n * n];
            for b in &blocks {
                for &i in b {
                    for &j in b {
                        same[i * n + j] = 1.0;
                    }
                }
            }
            let mut cls = layout.as_ref().unwrap().0.clone();
            for c in GRID_LABELS {
                let ci = order::class_id(c);
                for i in 0..n {
                    cls[i * nc + ci] = -1e4;
                }
            }
            layout = Some((cls, same));
        }
        // (table heads are off, as the Python default: no cell re-split pass)
        let note = if vertical { "order=xycut" } else { "order=model" };
        t.insert("order_ms".into(), ms(t0));
        let total: f64 = t.iter().filter(|(k, _)| *k != "total_ms").map(|(_, v)| v).sum();
        t.insert("total_ms".into(), total);
        let lines: Vec<Line> = ord
            .iter()
            .map(|&i| Line {
                bbox: [round1(boxes[i][0] as f64), round1(boxes[i][1] as f64), round1(boxes[i][2] as f64), round1(boxes[i][3] as f64)],
                text: rec[i].0.clone(),
                conf: ((rec[i].1 * 1e4).round_ties_even() / 1e4) as f32,
            })
            .collect();
        let mut md = None;
        let mut blocks = vec![];
        if let Some((cls, same)) = &layout {
            let gb: Vec<Bx> = boxes.iter().map(geo).collect();
            blocks = order::page_blocks(&gb, &texts, &ord, cls, same);
            for b in blocks.iter_mut() {
                let g = geo(&b.bbox);
                b.bbox = [order::round1(g[0]), order::round1(g[1]), order::round1(g[2]), order::round1(g[3])];
            }
            let mut cells: HashMap<usize, Vec<(Bx, String)>> = HashMap::new();
            for i in 0..boxes.len() {
                let parts = self.split_line(&boxes[i], &rec[i].2, CELL_GAP, &[], page_rtl);
                cells.insert(i, parts.into_iter().map(|(bx, tx, _)| (geo(&bx), tx)).collect());
            }
            let pics: Option<Vec<Bx>> = pictures.filter(|p| !p.is_empty()).map(|p| p.iter().map(geo).collect());
            md = Some(order::page_markdown(&gb, &texts, &ord, cls, same, self.furniture, &cells, pics.as_deref()));
            if let Some(p) = pictures.filter(|p| !p.is_empty()) {
                for pb in p {
                    let cy = (pb[1] + pb[3]) / 2.0;
                    let at = blocks.iter().position(|b| (b.bbox[1] + b.bbox[3]) / 2.0 > cy && pb[2].min(b.bbox[2]) > pb[0].max(b.bbox[0])).unwrap_or(blocks.len());
                    blocks.insert(at, Block { label: "Picture".into(), bbox: [order::round1(pb[0]), order::round1(pb[1]), order::round1(pb[2]), order::round1(pb[3])], text: String::new() });
                }
            }
        } else if boxes.len() < 2 {
            md = Some(texts.join("\n"));
        }
        // keep the borrow checker quiet about unused mutability on some paths
        let _ = (&mut boxes, &mut rec, &mut texts);
        Ok(Page { lines, markdown: md, blocks, source: "ocr".into(), note: note.into(), timings: t, size: wh, ..Default::default() })
    }

    /// Leader-only lines are dropped; a lone list marker joins its item.
    fn tidy_lines(&self, mut boxes: Vec<Bx>, rec: Vec<Rec>, rtl: bool) -> (Vec<Bx>, Vec<Rec>) {
        const LEADER: &str = ".·…‥_-–—=";
        const DASHES: &str = "-–—";
        const BULLETS: &str = "•◦▪▫●○■□‣⁃∙*";
        static ENUM: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let enumr = ENUM.get_or_init(|| regex::Regex::new(r"^(?:(\d{1,3}|[A-Za-z])[.)]|\((\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\))$").unwrap());
        let n = rec.len();
        let mut keep = vec![true; n];
        let mut rec = rec;
        let hs: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
        let ws: Vec<f32> = boxes.iter().map(|b| b[2] - b[0]).collect();
        for i in 0..n {
            let core: String = rec[i].0.chars().filter(|&c| c != ' ').collect();
            if !core.is_empty() && core.chars().all(|c| LEADER.contains(c)) {
                let wide = ws[i] >= 3.0 * hs[i];
                if core.chars().count() >= 3 || wide || !core.chars().any(|c| DASHES.contains(c)) {
                    keep[i] = false;
                }
            }
        }
        let space = self.vocab.space();
        for i in 0..n {
            let t = rec[i].0.trim_matches(text::is_space).to_string();
            let is_bullet = t.chars().count() == 1 && BULLETS.contains(t.chars().next().unwrap());
            if !keep[i] || !(is_bullet || enumr.is_match(&t)) {
                continue;
            }
            let (mut best, mut best_gap): (Option<usize>, Option<f32>) = (None, None);
            for j in 0..n {
                if j == i || !keep[j] || rec[j].0.trim_matches(text::is_space).is_empty() {
                    continue;
                }
                let ov = boxes[i][3].min(boxes[j][3]) - boxes[i][1].max(boxes[j][1]);
                if ov < 0.5 * hs[i].min(hs[j]) {
                    continue;
                }
                let gap = if rtl { boxes[i][0] - boxes[j][2] } else { boxes[j][0] - boxes[i][2] };
                if -0.3 * hs[i] <= gap && gap <= 2.5 * hs[i].max(hs[j]) && best_gap.map(|bg| gap < bg).unwrap_or(true) {
                    best = Some(j);
                    best_gap = Some(gap);
                }
            }
            let Some(bj) = best else { continue };
            let (tj, cj, mut kj) = rec[bj].clone();
            let chars = matches!(kj.first(), Some(Tok { v: TokV::Ch(_), .. }));
            let sp: Option<TokV> = if chars { Some(TokV::Ch(" ".into())) } else { space.map(TokV::Id) };
            let toks = rec[i].2.clone();
            if !rtl && !toks.is_empty() && !kj.is_empty() && sp.is_some() {
                let tchars = matches!(toks[0].v, TokV::Ch(_));
                if tchars == chars {
                    let x = (boxes[i][2] + boxes[bj][0]) as f64 / 2.0;
                    let mut nk = toks.clone();
                    nk.push(Tok { v: sp.unwrap(), x, seq: toks.last().unwrap().seq });
                    nk.extend(kj);
                    kj = nk;
                }
            }
            rec[bj] = (format!("{t} {}", tj.trim_matches(text::is_space)), cj, kj);
            boxes[bj] = [boxes[i][0].min(boxes[bj][0]), boxes[i][1].min(boxes[bj][1]), boxes[i][2].max(boxes[bj][2]), boxes[i][3].max(boxes[bj][3])];
            keep[i] = false;
        }
        let idx: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
        (idx.iter().map(|&i| boxes[i]).collect(), idx.iter().map(|&i| rec[i].clone()).collect())
    }
}
