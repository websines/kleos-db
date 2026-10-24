//! Whole documents (`document.py`): the PDF text layer where it can be
//! trusted, OCR elsewhere; images straight to OCR.

use std::collections::BTreeMap;
use std::path::Path;

use super::cv::{self, Gray};
use super::models::ModelError;
use super::pipeline::{Ocr, Page, Rec, Tok, TokV};
use super::text::{self, is_alpha};
use super::textlayer::{self, read_text_layer, TextLayer, TextLine, BROKEN_FONT_MAP};
use crate::pdf::{Doc, PdfError};

const CHECK_DPI: f64 = 72.0;
const MAX_PAGE_PIXELS: f64 = 100e6;
const READABLE_MIN: f64 = 0.9;
const AGREE_LINES: usize = 6;
const AGREE_SHARE: f64 = 0.25;
const LATIN_CODES: f64 = 0.9;
const NOT_THE_PAGE: &str = "text layer does not match the page";

#[derive(Debug, thiserror::Error)]
pub enum DocError {
    #[error(transparent)]
    Pdf(#[from] PdfError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("not a PDF or a readable image: {0}")]
    Input(String),
}

/// Text-layer handling: auto (layer where sound), ocr (never the layer), text (layer even when checks fail).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Auto,
    Ocr,
    Text,
}

fn latin_share(t: &str) -> f64 {
    let letters: Vec<char> = t.chars().filter(|&c| is_alpha(c)).collect();
    letters.iter().filter(|c| c.is_ascii()).count() as f64 / letters.len().max(1) as f64
}

fn readable(ocr: &Ocr, runs: &[TextLine]) -> bool {
    let letters: Vec<char> = runs.iter().flat_map(|r| r.text.chars()).filter(|&c| is_alpha(c)).collect();
    if letters.is_empty() {
        return true;
    }
    use unicode_normalization::UnicodeNormalization;
    let known = letters
        .iter()
        .filter(|&&c| {
            let s = c.to_string();
            ocr.knows(&s) || s.nfd().all(|j| ocr.knows(&j.to_string()))
        })
        .count();
    known as f64 >= READABLE_MIN * letters.len() as f64
}

fn right_to_left(toks: &[Tok]) -> bool {
    let (mut r, mut l) = (0, 0);
    for t in toks {
        if let TokV::Ch(c) = &t.v {
            let mut it = c.chars();
            if let (Some(ch), None) = (it.next(), it.next()) {
                r += text::is_rtl(ch) as usize;
                l += text::bidi_l(ch) as usize;
            }
        }
    }
    r > l
}

fn covered(b: &[f32; 4], boxes: &[[f32; 4]], frac: f32) -> bool {
    let area = ((b[2] - b[0]) * (b[3] - b[1])).max(1e-6);
    boxes.iter().any(|o| {
        let ix = (b[2].min(o[2]) - b[0].max(o[0])).max(0.0);
        let iy = (b[3].min(o[3]) - b[1].max(o[1])).max(0.0);
        ix * iy / area > frac
    })
}

/// Tokens (char, x, content index) per detected line; characters outside every box.
fn assign_runs(runs: &[TextLine], boxes: &[[f32; 4]], k: f64) -> (Vec<Vec<Tok>>, usize) {
    if boxes.is_empty() {
        return (vec![], runs.iter().flat_map(|r| r.chars.iter()).filter(|(c, _)| !text::is_space(*c)).count());
    }
    let area: Vec<f32> = boxes.iter().map(|b| (b[2] - b[0]) * (b[3] - b[1])).collect();
    let mut parts: Vec<BTreeMap<i64, Vec<Tok>>> = vec![BTreeMap::new(); boxes.len()];
    let mut first_pos: Vec<Vec<i64>> = vec![vec![]; boxes.len()]; // insertion order of run indices
    let mut lost = 0;
    for (pos, run) in runs.iter().enumerate() {
        let idx = run.seq.unwrap_or(pos as i64);
        let mut j: Option<usize> = None;
        for (c, b) in &run.chars {
            if !text::is_space(*c) {
                let cx = ((b[0] + b[2]) / 2.0 * k) as f32;
                let cy = ((b[1] + b[3]) / 2.0 * k) as f32;
                let mut best: Option<usize> = None;
                for (bi, bx) in boxes.iter().enumerate() {
                    if bx[0] <= cx && cx <= bx[2] && bx[1] <= cy && cy <= bx[3] && best.map(|b0| area[bi] < area[b0]).unwrap_or(true) {
                        best = Some(bi);
                    }
                }
                let Some(bj) = best else {
                    lost += 1;
                    j = None;
                    continue;
                };
                j = Some(bj);
                if !parts[bj].contains_key(&idx) {
                    first_pos[bj].push(idx);
                }
                parts[bj].entry(idx).or_default().push(Tok { v: TokV::Ch(c.to_string()), x: cx as f64, seq: Some(idx) });
            } else if let Some(bj) = j {
                let prev = parts[bj][&idx].last().unwrap().x;
                parts[bj].get_mut(&idx).unwrap().push(Tok { v: TokV::Ch(c.to_string()), x: prev, seq: Some(idx) });
            }
        }
    }
    let mut toks = vec![];
    for (li, line) in parts.iter().enumerate() {
        let all: Vec<Tok> = first_pos[li].iter().flat_map(|i| line[i].clone()).collect();
        let rtl = right_to_left(&all);
        let mut order = first_pos[li].clone();
        let key_max = |i: &i64| line[i].iter().map(|t| t.x).fold(f64::NEG_INFINITY, f64::max);
        let key_min = |i: &i64| line[i].iter().map(|t| t.x).fold(f64::INFINITY, f64::min);
        if rtl {
            order.sort_by(|a, b| key_max(b).partial_cmp(&key_max(a)).unwrap());
        } else {
            order.sort_by(|a, b| key_min(a).partial_cmp(&key_min(b)).unwrap());
        }
        let mut tk: Vec<Tok> = vec![];
        for idx in order {
            if let Some(last) = tk.last() {
                let x = last.x;
                tk.push(Tok { v: TokV::Ch(" ".into()), x, seq: Some(idx) });
            }
            tk.extend(line[&idx].iter().cloned());
        }
        toks.push(tk);
    }
    (toks, lost)
}

/// A piece of a text-layer run on one visual line: (run, char range, box in points).
type Seg = (usize, std::ops::Range<usize>, [f64; 4]);

/// Runs cut where their characters leave the current line (a run can carry a
/// hyphenated word across a line break), then merged into visual lines:
/// (segments left to right, union box in points), top to bottom.  Segments
/// join a line when they share most of its vertical band.
fn merge_runs(runs: &[TextLine]) -> Vec<(Vec<Seg>, [f64; 4])> {
    let mut segs: Vec<Seg> = vec![];
    for (ri, run) in runs.iter().enumerate() {
        let mut cur: Option<(usize, [f64; 4])> = None;
        for (ci, (c, b)) in run.chars.iter().enumerate() {
            if text::is_space(*c) {
                continue;
            }
            match &mut cur {
                Some((st, bb)) => {
                    let (h, cy) = (bb[3] - bb[1], (b[1] + b[3]) / 2.0);
                    let back = b[0] < bb[2] - 2.0 * h.max(1.0);
                    if cy < bb[1] || cy > bb[3] || back {
                        segs.push((ri, *st..ci, *bb));
                        cur = Some((ci, *b));
                    } else {
                        *bb = [bb[0].min(b[0]), bb[1].min(b[1]), bb[2].max(b[2]), bb[3].max(b[3])];
                    }
                }
                None => cur = Some((ci, *b)),
            }
        }
        if let Some((st, bb)) = cur {
            segs.push((ri, st..run.chars.len(), bb));
        }
    }
    segs.sort_by(|a, b| {
        let (ya, yb) = ((a.2[1] + a.2[3]) / 2.0, (b.2[1] + b.2[3]) / 2.0);
        ya.partial_cmp(&yb).unwrap().then(a.2[0].partial_cmp(&b.2[0]).unwrap())
    });
    let mut lines: Vec<(Vec<Seg>, [f64; 4])> = vec![];
    for sg in segs {
        let b = sg.2;
        let h = b[3] - b[1];
        let cy = (b[1] + b[3]) / 2.0;
        let joined = lines.iter_mut().rev().take(3).find(|(_, lb)| {
            let lh = lb[3] - lb[1];
            let ov = lb[3].min(b[3]) - lb[1].max(b[1]);
            // a small mark (comma, quote, footnote) joins the band it touches
            ov > 0.6 * h.min(lh) || (h < 0.6 * lh && cy >= lb[1] - 0.3 * lh && cy <= lb[3] + 0.3 * lh)
        });
        match joined {
            Some((ids, lb)) => {
                *lb = [lb[0].min(b[0]), lb[1].min(b[1]), lb[2].max(b[2]), lb[3].max(b[3])];
                ids.push(sg);
            }
            None => lines.push((vec![sg.clone()], b)),
        }
    }
    for (ids, _) in lines.iter_mut() {
        ids.sort_by(|a, b| a.2[0].partial_cmp(&b.2[0]).unwrap());
    }
    lines
}

/// Read an image file as grayscale (OpenCV's RGB weights).
#[cfg(feature = "ocr")]
pub fn read_image(bytes: &[u8]) -> Result<Gray, DocError> {
    let img = image::load_from_memory(bytes).map_err(|e| DocError::Input(e.to_string()))?;
    Ok(match img {
        image::DynamicImage::ImageLuma8(g) => {
            let (w, h) = g.dimensions();
            Gray::from_raw(w as usize, h as usize, g.into_raw())
        }
        other => {
            let rgb = other.to_rgb8();
            let (w, h) = rgb.dimensions();
            let data = rgb.pixels().map(|p| cv::rgb_to_gray(p[0], p[1], p[2])).collect();
            Gray::from_raw(w as usize, h as usize, data)
        }
    })
}

pub fn is_pdf(bytes: &[u8]) -> bool {
    bytes.windows(5).take(1024).any(|w| w == b"%PDF-")
}

/// What a PDF page needs from pdfium (see `Converter::prepare`).
pub struct Prepared {
    pub size: (f64, f64),
    pub scale: f64,
    /// a sideways page read with its /Rotate removed: (layer, render, scale)
    pub upright: Option<(TextLayer, Gray, f64)>,
    pub layer: Option<TextLayer>,
    /// the page rendered at `scale`
    pub gray: Gray,
    pub text_ms: f64,
    pub render_ms: f64,
    /// a simple digital page read from its text layer's geometry alone (no
    /// render, no detector): see `Converter::simple_page`
    pub fast: bool,
}

/// Fast-path limits: a page with more vector drawing or image area than this
/// goes through the detector (tables, figures, mixed pages).
const FAST_MAX_PATHS: usize = 40;
const FAST_MAX_IMAGE_SHARE: f64 = 0.01;

pub struct Converter<'a> {
    pub ocr: &'a Ocr,
    /// The engine for full-OCR pages (default: `ocr` itself, Indra).
    pub backend: Option<&'a dyn super::OcrBackend>,
    pub dpi: f64,
    pub mode: Mode,
    pub image_ocr: bool,
    /// Read simple digital pages from the text layer's geometry, skipping the
    /// render and the line detector (`KLEOS_DOC_FAST=0` turns it off).
    pub fast_digital: bool,
}

impl<'a> Converter<'a> {
    pub fn new(ocr: &'a Ocr) -> Self {
        let fast_digital = std::env::var("KLEOS_DOC_FAST").map(|v| v != "0").unwrap_or(true);
        Converter { ocr, backend: None, dpi: 200.0, mode: Mode::Auto, image_ocr: true, fast_digital }
    }

    fn page_scale(&self, size: (f64, f64)) -> f64 {
        let k = self.dpi / 72.0;
        k.min((MAX_PAGE_PIXELS / (size.0 * size.1).max(1.0)).sqrt())
    }

    /// Everything a page needs from pdfium (renders and text layers), read
    /// under the library lock; the OCR work then runs without it.
    pub fn prepare(&self, doc: &Doc, index: usize) -> Result<Prepared, DocError> {
        Ok(doc.with_page(index, |page| {
            let size = page.size();
            let k = self.page_scale(size);
            let mut p = Prepared { size, scale: k, upright: None, layer: None, gray: Gray::new(1, 1, 255), text_ms: 0.0, render_ms: 0.0, fast: false };
            if self.mode != Mode::Ocr {
                let t0 = std::time::Instant::now();
                let s = CHECK_DPI / 72.0;
                // the 72-dpi ink-check render is made only when a check needs it
                let check_cell: std::cell::OnceCell<Gray> = std::cell::OnceCell::new();
                let check = || check_cell.get_or_init(|| page.render_gray(s)).clone();
                let mut tl = read_text_layer(page, &check, s, 0, None);
                let kq = textlayer::page_turn(&tl.stats);
                let rot = page.rotation();
                if kq != 0 && (rot + 90 * kq) % 360 == 0 {
                    page.set_rotation(0);
                    let up = || page.render_gray(s);
                    let mut tl0 = read_text_layer(page, &up, s, 0, None);
                    if !tl0.lines.is_empty() && (tl0.trusted || self.mode == Mode::Text) {
                        tl0.stats.turn = kq;
                        let g = page.render_gray(self.page_scale(page.size()));
                        p.upright = Some((tl0, g, self.page_scale(page.size())));
                    }
                    page.set_rotation(rot);
                    tl = read_text_layer(page, &|| cv::rot90(&check(), -kq), s, kq, None);
                } else if kq != 0 {
                    tl = read_text_layer(page, &|| cv::rot90(&check(), -kq), s, kq, None);
                } else if textlayer::vertical_layout(&tl.stats) {
                    tl = read_text_layer(page, &|| cv::rot90(&check(), 1), s, 0, Some(3));
                    tl.stats.vertical = true;
                }
                p.text_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if self.fast_digital && self.mode == Mode::Auto && p.upright.is_none() && self.simple_page(&tl, size, &page.census()) {
                    p.layer = Some(tl);
                    p.fast = true;
                    return p;
                }
                p.layer = Some(tl);
            }
            let t0 = std::time::Instant::now();
            p.gray = page.render_gray(k);
            p.render_ms = t0.elapsed().as_secs_f64() * 1000.0;
            p
        })?)
    }

    /// One PDF page.
    pub fn pdf_page(&self, doc: &Doc, index: usize) -> Result<Page, DocError> {
        let p = self.prepare(doc, index)?;
        self.finish(p)
    }

    /// The page from what `prepare` read: the text layer where sound, OCR elsewhere.
    pub fn finish(&self, p: Prepared) -> Result<Page, DocError> {
        let mut note = "mode=ocr".to_string();
        let mut turn = true;
        let stamp = |mut out: Page, p: &Prepared| {
            out.timings.insert("text_ms".into(), p.text_ms);
            *out.timings.entry("render_ms".into()).or_insert(0.0) += p.render_ms;
            let total: f64 = out.timings.iter().filter(|(k, _)| *k != "total_ms").map(|(_, v)| v).sum();
            out.timings.insert("total_ms".into(), total);
            out
        };
        if p.fast {
            if let Some(tl) = &p.layer {
                return Ok(stamp(self.geometry_page(tl, p.size, p.scale)?, &p));
            }
        }
        if let Some((tl0, g0, k0)) = &p.upright {
            if let Some(out) = self.text_page(tl0, g0.clone(), *k0, self.mode == Mode::Auto)? {
                return Ok(stamp(out, &p));
            }
        }
        if let Some(tl) = &p.layer {
            let unreadable = tl.reason != BROKEN_FONT_MAP && !readable(self.ocr, &tl.lines);
            note = tl.reason.clone();
            if !tl.lines.is_empty() && (tl.trusted || self.mode == Mode::Text || unreadable) {
                if let Some(out) = self.text_page(tl, p.gray.clone(), p.scale, self.mode == Mode::Auto && tl.trusted && !unreadable)? {
                    return Ok(stamp(out, &p));
                }
                note = NOT_THE_PAGE.into();
            }
            let st = &tl.stats;
            turn = !(tl.reason == BROKEN_FONT_MAP && (st.rotated as f64) < 0.5 * st.chars as f64);
        }
        let mut out = match self.backend {
            Some(b) => b.read_page(&p.gray, turn).map_err(DocError::Input)?,
            None => self.ocr.read(&p.gray, turn)?,
        };
        out.note = format!("{note}; {}", out.note);
        Ok(stamp(out, &p))
    }

    /// Whether a digital page can be read from its text layer's geometry alone:
    /// a trusted, upright, horizontal, Latin, readable layer; next to no images
    /// or vector drawing (tables, figures); and one column (no two lines side
    /// by side on the same band).
    fn simple_page(&self, tl: &TextLayer, size: (f64, f64), census: &crate::pdf::Census) -> bool {
        let st = &tl.stats;
        if !tl.trusted || tl.lines.is_empty() || st.turn != 0 || st.frame != 0 || st.vertical || !tl.rotated.is_empty() {
            return false;
        }
        if census.paths > FAST_MAX_PATHS || census.shadings > 0 || census.image_area > FAST_MAX_IMAGE_SHARE * size.0 * size.1 {
            return false;
        }
        let all: String = tl.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" ");
        if latin_share(&all) < LATIN_CODES || !readable(self.ocr, &tl.lines) {
            return false;
        }
        let lines = merge_runs(&tl.lines);
        // prose: most lines share the left margin and run most of the width
        // (forms, tables and lists do not)
        let n = lines.len();
        let mut hs: Vec<f64> = lines.iter().map(|(_, b)| b[3] - b[1]).collect();
        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mh = hs[n / 2].max(1.0);
        let left = lines.iter().map(|(_, a)| lines.iter().filter(|(_, b)| (a[0] - b[0]).abs() < mh).count()).max().unwrap_or(0);
        let maxw = lines.iter().map(|(_, b)| b[2] - b[0]).fold(0.0, f64::max);
        let wide = lines.iter().filter(|(_, b)| b[2] - b[0] >= 0.5 * maxw).count();
        if n < 4 || (left as f64) < 0.6 * n as f64 || (wide as f64) < 0.6 * n as f64 {
            return false;
        }
        // a wide gap inside a band: columns or table cells
        for (ids, lb) in &lines {
            let hgt = lb[3] - lb[1];
            for pair in ids.windows(2) {
                if pair[1].2[0] - pair[0].2[2] > 2.0 * hgt {
                    return false;
                }
            }
        }
        for (i, a) in lines.iter().enumerate() {
            for b in lines.iter().skip(i + 1) {
                let (ha, hb) = (a.1[3] - a.1[1], b.1[3] - b.1[1]);
                let ov = a.1[3].min(b.1[3]) - a.1[1].max(b.1[1]);
                if ov > 0.3 * ha.min(hb) && (a.1[2] <= b.1[0] || b.1[2] <= a.1[0]) {
                    return false;
                }
            }
        }
        true
    }

    /// A simple digital page from its text layer: lines are the layer's runs
    /// merged per baseline band, in the same token form the detector path
    /// builds, then the usual order, layout and Markdown.
    fn geometry_page(&self, tl: &TextLayer, size: (f64, f64), k: f64) -> Result<Page, DocError> {
        let mut t = std::collections::HashMap::new();
        t.insert("det_ms".to_string(), 0.0);
        let (w, h) = (((size.0 * k).ceil() as usize).max(1), ((size.1 * k).ceil() as usize).max(1));
        let mut kb: Vec<[f32; 4]> = vec![];
        let mut kr: Vec<Rec> = vec![];
        for (segs, bb) in merge_runs(&tl.lines) {
            let mut tk: Vec<Tok> = vec![];
            let mut last_idx: Option<i64> = None;
            for (ri, range, _) in &segs {
                let run = &tl.lines[*ri];
                let idx = run.seq.unwrap_or(*ri as i64);
                // as the detector path: pieces of one content object join
                // without a space; a space separates different objects
                let mut started = last_idx == Some(idx);
                last_idx = Some(idx);
                for (c, b) in &run.chars[range.clone()] {
                    if text::is_space(*c) {
                        if started {
                            let x = tk.last().map(|t| t.x).unwrap_or(0.0);
                            tk.push(Tok { v: TokV::Ch(c.to_string()), x, seq: Some(idx) });
                        }
                        continue;
                    }
                    if !started && !tk.is_empty() {
                        let x = tk.last().unwrap().x;
                        tk.push(Tok { v: TokV::Ch(" ".into()), x, seq: Some(idx) });
                    }
                    started = true;
                    tk.push(Tok { v: TokV::Ch(c.to_string()), x: (b[0] + b[2]) / 2.0 * k, seq: Some(idx) });
                }
            }
            let txt = if tk.is_empty() { String::new() } else { self.ocr.decode_toks(&tk, false) };
            if txt.trim().is_empty() {
                continue;
            }
            kb.push([(bb[0] * k) as f32, (bb[1] * k) as f32, (bb[2] * k) as f32, (bb[3] * k) as f32]);
            kr.push((txt, 1.0, tk));
        }
        let mut out = self.ocr.assemble((w, h), kb, kr, t, false, None)?;
        out.source = "text".into();
        out.rotation = 0;
        out.size = (w, h);
        out.note += "; geometry fast path";
        Ok(out)
    }

    /// The page from its text layer (None: the layer does not match the page, OCR it).
    fn text_page(&self, tl: &TextLayer, gray: Gray, k: f64, check: bool) -> Result<Option<Page>, DocError> {
        let turn = tl.stats.turn;
        let frame = tl.stats.frame;
        let vertical = tl.stats.vertical;
        let mut t = std::collections::HashMap::new();
        let gray = if frame != 0 { cv::rot90(&gray, -frame) } else { gray };
        let t0 = std::time::Instant::now();
        let det = self.ocr.detect(&gray)?;
        t.insert("det_ms".to_string(), t0.elapsed().as_secs_f64() * 1000.0);
        let boxes: Vec<[f32; 4]> = det.iter().map(|b| [b[0], b[1], b[2], b[3]]).collect();
        let (toks, lost) = assign_runs(&tl.lines, &boxes, k);
        let mut rec: Vec<Option<Rec>> = vec![None; boxes.len()];
        let mut empty = vec![];
        for (i, tk) in toks.iter().enumerate() {
            let txt = if tk.is_empty() { String::new() } else { self.ocr.decode_toks(tk, false) };
            if !txt.is_empty() {
                rec[i] = Some((txt, 1.0, tk.clone()));
            } else {
                empty.push(i);
            }
        }
        if check {
            let full: Vec<usize> = (0..rec.len()).filter(|&i| rec[i].as_ref().map(|r| r.0.chars().filter(|&c| is_alpha(c)).count() >= 4).unwrap_or(false)).collect();
            let step = (full.len() / AGREE_LINES).max(1);
            let pick: Vec<usize> = full.iter().step_by(step).take(AGREE_LINES).copied().collect();
            if !pick.is_empty() {
                let pb: Vec<_> = pick.iter().map(|&i| det[i]).collect();
                let (groups, sure) = self.ocr.box_groups(&gray, &pb)?;
                let agree: Vec<bool> = pick.iter().zip(groups.iter().zip(sure.iter())).map(|(&i, (g, &s))| !(latin_share(&rec[i].as_ref().unwrap().0) >= LATIN_CODES && g != "base" && s)).collect();
                if !(agree.is_empty() || agree.iter().filter(|&&a| a).count() as f64 >= AGREE_SHARE * agree.len() as f64) {
                    return Ok(None);
                }
            }
        }
        let mut source = "text";
        let side: Vec<&TextLine> = tl.rotated.iter().filter(|r| !r.text.trim().is_empty()).collect();
        let side_boxes: Vec<[f32; 4]> = side.iter().map(|r| [(r.bbox[0] * k) as f32, (r.bbox[1] * k) as f32, (r.bbox[2] * k) as f32, (r.bbox[3] * k) as f32]).collect();
        if !side.is_empty() && !empty.is_empty() {
            empty.retain(|&i| !covered(&boxes[i], &side_boxes, 0.5));
        }
        if !empty.is_empty() && self.image_ocr && readable(self.ocr, &tl.lines) {
            let t0 = std::time::Instant::now();
            let eb: Vec<_> = empty.iter().map(|&i| det[i]).collect();
            for (&i, r) in empty.iter().zip(self.ocr.recognize(&gray, &eb, vertical)?) {
                if !r.0.trim().is_empty() {
                    rec[i] = Some(r);
                    source = "hybrid";
                }
            }
            t.insert("rec_ms".to_string(), t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut kb: Vec<[f32; 4]> = vec![];
        let mut kr: Vec<Rec> = vec![];
        for (i, r) in rec.into_iter().enumerate() {
            if let Some(r) = r {
                kb.push(boxes[i]);
                kr.push(r);
            }
        }
        for (r, b) in side.iter().zip(side_boxes.iter()) {
            let cx = (b[0] + b[2]) as f64 / 2.0;
            kb.push(*b);
            kr.push((r.text.clone(), 1.0, r.text.chars().map(|c| Tok { v: TokV::Ch(c.to_string()), x: cx, seq: r.seq }).collect()));
        }
        let mut out = self.ocr.assemble((gray.w, gray.h), kb, kr, t, vertical, None)?;
        out.source = source.into();
        out.rotation = 90 * turn;
        out.size = (gray.w, gray.h);
        if vertical {
            let w = gray.h as f32;
            let back = |b: [f32; 4]| [w - b[3], b[0], w - b[1], b[2]];
            for l in out.lines.iter_mut() {
                l.bbox = back(l.bbox);
            }
            for b in out.blocks.iter_mut() {
                b.bbox = back(b.bbox);
            }
            out.vertical = true;
        }
        if lost > 0 {
            out.note += &format!("; {lost} chars outside detected lines");
        }
        Ok(Some(out))
    }

    /// Pages of a PDF or image given as bytes.  `max_pages` 0 reads all.
    pub fn convert_bytes(&self, bytes: Vec<u8>, max_pages: usize, mut each: impl FnMut(usize, Page)) -> Result<usize, DocError> {
        if !is_pdf(&bytes) {
            let g = read_image(&bytes)?;
            each(0, self.safe(|| Ok(self.ocr.read(&g, true)?)));
            return Ok(1);
        }
        let doc = Doc::from_bytes(bytes, None)?;
        self.convert_doc(&doc, 0, max_pages, &mut each)
    }

    pub fn convert_path(&self, path: &Path, max_pages: usize, mut each: impl FnMut(usize, Page)) -> Result<usize, DocError> {
        let head = std::fs::read(path).map_err(|e| DocError::Input(format!("{}: {e}", path.display())))?;
        if !is_pdf(&head[..head.len().min(1024)]) {
            let g = read_image(&head)?;
            each(0, self.safe(|| Ok(self.ocr.read(&g, true)?)));
            return Ok(1);
        }
        let doc = Doc::open(path, None)?;
        self.convert_doc(&doc, 0, max_pages, &mut each)
    }

    /// Pages [from, from + max) of an open PDF.
    pub fn convert_doc(&self, doc: &Doc, from: usize, max_pages: usize, each: &mut impl FnMut(usize, Page)) -> Result<usize, DocError> {
        let end = if max_pages == 0 { doc.pages } else { (from + max_pages).min(doc.pages) };
        for i in from..end {
            each(i, self.safe(|| self.pdf_page(doc, i)));
        }
        Ok(end.saturating_sub(from))
    }

    /// Pages [from, end) on `threads` workers: pdfium reads run one at a
    /// time (its lock), OCR in parallel.  `each` is called in page order.
    pub fn convert_doc_parallel(&self, doc: &Doc, from: usize, end: usize, threads: usize, each: &mut (dyn FnMut(usize, Page) + Send)) -> usize {
        self.convert_doc_streaming(doc, from, end, threads, &std::sync::atomic::AtomicBool::new(false), each)
    }

    /// As `convert_doc_parallel`, for a long stream: workers stop taking pages
    /// once `cancel` is set, and they wait (bounded queue) while `each` is busy,
    /// so a slow consumer holds the workers back instead of buffering the book.
    pub fn convert_doc_streaming(&self, doc: &Doc, from: usize, end: usize, threads: usize, cancel: &std::sync::atomic::AtomicBool, each: &mut (dyn FnMut(usize, Page) + Send)) -> usize {
        let end = end.min(doc.pages);
        if end <= from {
            return 0;
        }
        let next = std::sync::atomic::AtomicUsize::new(from);
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Page)>(2 * threads.max(1));
        std::thread::scope(|s| {
            for wi in 0..threads.max(1) {
                let tx = tx.clone();
                let next = &next;
                let w = wi;
                s.spawn(move || {
                    // each worker reads its pages through its own pdfium
                    // instance (a separate copy of the library), so renders
                    // and text layers run in parallel
                    let own = if w == 0 { None } else { doc.reopen_on(w).ok() };
                    let d = own.as_ref().unwrap_or(doc);
                    loop {
                        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if i >= end {
                            break;
                        }
                        let page = self.safe(|| self.pdf_page(d, i));
                        if tx.send((i, page)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(tx);
            let mut pending: BTreeMap<usize, Page> = BTreeMap::new();
            let mut want = from;
            for (i, p) in rx {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    continue; // drain so the workers can finish
                }
                pending.insert(i, p);
                while let Some(p) = pending.remove(&want) {
                    each(want, p);
                    want += 1;
                }
            }
        });
        end - from
    }

    fn safe(&self, f: impl FnOnce() -> Result<Page, DocError>) -> Page {
        // one bad page (an error, or a panic in a model) must not end the document
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => Page { markdown: Some(String::new()), source: "error".into(), note: e.to_string(), ..Default::default() },
            Err(p) => {
                let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default();
                Page { markdown: Some(String::new()), source: "error".into(), note: format!("panic: {msg}"), ..Default::default() }
            }
        }
    }
}
