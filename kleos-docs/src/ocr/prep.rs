//! Detector input and decoding (`det_post.py`) and recognizer line crops and
//! batches (`rec_preproc.py`, `line_mask.py`).

use super::cv::{self, Gray};

pub const CORE_X_SHRINK: f32 = 0.15;
pub const OUT_STRIDE: f32 = 2.0;
pub const INPUT_MULTIPLE: usize = 32;

pub const LINE_HEIGHT: usize = 32;
pub const PAD_Y: f32 = 0.2;
pub const PAD_X: f32 = 0.3;
pub const WIDTH_MULTIPLE: usize = 8;

/// A box with a score: x0, y0, x1, y1, score.
pub type DetBox = [f32; 5];

/// Page resized so its long side is `long_side`, padded white to a multiple of
/// 32: (ink map [H * W], H, W, scale).
pub fn page_to_input(gray: &Gray, long_side: usize) -> (Vec<f32>, usize, usize, f64) {
    let (h, w) = (gray.h, gray.w);
    let scale = long_side as f64 / h.max(w) as f64;
    let nh = ((h as f64 * scale).round_ties_even() as usize).max(1);
    let nw = ((w as f64 * scale).round_ties_even() as usize).max(1);
    let g = if (nh, nw) != (h, w) {
        if scale < 1.0 {
            cv::resize_area(gray, nw, nh)
        } else {
            cv::resize_linear(gray, nw, nh)
        }
    } else {
        gray.clone()
    };
    let ph = nh.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let pw = nw.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let mut out = vec![0f32; ph * pw];
    for y in 0..nh {
        for x in 0..nw {
            out[y * pw + x] = 1.0 - g.data[y * nw + x] as f32 / 255.0;
        }
    }
    (out, ph, pw, scale)
}

/// Decode one page of detector output [3 or 4, h, w] (core logit, top
/// distance, bottom distance, ...) into boxes in input pixels.
pub fn decode(out: &[f32], h: usize, w: usize) -> Vec<DetBox> {
    let n = h * w;
    let core = &out[..n];
    let top = &out[n..2 * n];
    let bot = &out[2 * n..3 * n];
    let prob: Vec<f32> = core.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    let mask: Vec<bool> = prob.iter().map(|&p| p > 0.5).collect();
    let (lab, stats) = cv::connected_components(&mask, w, h, false);
    let nl = stats.len();
    if nl <= 1 {
        return vec![];
    }
    let mut cnt = vec![0f64; nl];
    let mut y0 = vec![0f64; nl];
    let mut y1 = vec![0f64; nl];
    let mut sc = vec![0f64; nl];
    for y in 0..h {
        let ys = (y as f32 + 0.5) * OUT_STRIDE;
        for x in 0..w {
            let i = y * w + x;
            let l = lab[i] as usize;
            if l == 0 {
                continue;
            }
            cnt[l] += 1.0;
            y0[l] += (ys - top[i]) as f64;
            y1[l] += (ys + bot[i]) as f64;
            sc[l] += prob[i] as f64;
        }
    }
    let mut boxes = vec![];
    for i in 1..nl {
        if stats[i].area < 2 {
            continue;
        }
        let c = cnt[i].max(1.0);
        let (a, b, s) = ((y0[i] / c) as f32, (y1[i] / c) as f32, (sc[i] / c) as f32);
        let lh = (b - a).max(1.0);
        let ex = (CORE_X_SHRINK * lh).min(0.5 * lh);
        let x0 = stats[i].left as f32 * OUT_STRIDE - ex;
        let x1 = (stats[i].left + stats[i].width) as f32 * OUT_STRIDE + ex;
        boxes.push([x0, a, x1, b, s]);
    }
    boxes
}

fn percentiles(b: &Gray) -> (f64, f64, f64) {
    let mut hist = [0u64; 256];
    for &v in &b.data {
        hist[v as usize] += 1;
    }
    let n = b.data.len() as u64;
    (cv::percentile_u8(&hist, n, 0.02), cv::percentile_u8(&hist, n, 0.5), cv::percentile_u8(&hist, n, 0.98))
}

/// Light text on a dark ground (read inverted).
pub fn is_dark_line(gray: &Gray, b: &[f32]) -> bool {
    let (x0, y0, x1, y1) = (b[0], b[1], b[2], b[3]);
    let ys = (y0.max(0.0) as i64).max(0) as usize;
    let xs = (x0.max(0.0) as i64).max(0) as usize;
    let ye = ((y1 as i64) + 1).clamp(0, gray.h as i64) as usize;
    let xe = ((x1 as i64) + 1).clamp(0, gray.w as i64) as usize;
    // Python int() truncates toward zero; boxes are clipped to the page
    let ys = ys.min(gray.h);
    let xs = xs.min(gray.w);
    if ye <= ys || xe <= xs {
        return false;
    }
    let c = gray.crop(xs, ys, xe, ye);
    let (lo, med, hi) = percentiles(&c);
    hi - med > 2.0 * (med - lo) + 8.0
}

/// One line crop: a float ink map [window * height, W] (row-major), W.
pub fn crop_line(gray: &Gray, b: &[f32], height: usize, window: usize, dark: Option<bool>) -> (Vec<f32>, usize) {
    let pad_y = (window as f64 * (1.0 + 2.0 * PAD_Y as f64) - 1.0) / 2.0;
    let height = height * window;
    let (x0, y0, x1, y1) = (b[0] as f64, b[1] as f64, b[2] as f64, b[3] as f64);
    let h = (y1 - y0).max(1.0);
    let (xx0, xx1) = (x0 - PAD_X as f64 * h, x1 + PAD_X as f64 * h);
    let (yy0, yy1) = (y0 - pad_y * h, y1 + pad_y * h);
    let (hh, ww) = (gray.h as i64, gray.w as i64);
    let (ix0, iy0) = (xx0.floor() as i64, yy0.floor() as i64);
    let (ix1, iy1) = (xx1.ceil() as i64, yy1.ceil() as i64);
    let cw = (ix1 - ix0).max(1);
    let ch = (iy1 - iy0).max(1);
    let (sx0, sy0, sx1, sy1) = (ix0.max(0), iy0.max(0), ix1.min(ww), iy1.min(hh));
    let dark = dark.unwrap_or_else(|| is_dark_line(gray, b));
    let mut crop = Gray::new(cw as usize, ch as usize, if dark { 0 } else { 255 });
    if sx1 > sx0 && sy1 > sy0 {
        for y in sy0..sy1 {
            let src = &gray.data[(y * ww + sx0) as usize..(y * ww + sx1) as usize];
            let d0 = ((y - iy0) * cw + (sx0 - ix0)) as usize;
            crop.data[d0..d0 + src.len()].copy_from_slice(src);
        }
    }
    let out_w = ((cw as f64 * height as f64 / ch as f64).round_ties_even() as usize).clamp(WIDTH_MULTIPLE, 4096).max(WIDTH_MULTIPLE);
    let resized = if ch as usize > height { cv::resize_area(&crop, out_w, height) } else { cv::resize_linear(&crop, out_w, height) };
    let ink = resized.data.iter().map(|&v| {
        let f = v as f32 / 255.0;
        if dark {
            f
        } else {
            1.0 - f
        }
    });
    (ink.collect(), out_w)
}

/// Right-pad ink maps [H, W_i] to one width: (batch [B, 1, H, Wm], widths, Wm).
pub fn pad_batch(lines: &[&(Vec<f32>, usize)], height: usize) -> (Vec<f32>, Vec<i64>, usize) {
    let wm = lines.iter().map(|l| l.1).max().unwrap_or(WIDTH_MULTIPLE);
    let wm = wm.div_ceil(WIDTH_MULTIPLE) * WIDTH_MULTIPLE;
    let mut out = vec![0f32; lines.len() * height * wm];
    let mut widths = vec![0i64; lines.len()];
    for (i, (d, w)) in lines.iter().map(|l| (&l.0, l.1)).enumerate() {
        for y in 0..height {
            let o = i * height * wm + y * wm;
            out[o..o + w].copy_from_slice(&d[y * w..(y + 1) * w]);
        }
        widths[i] = w as i64;
    }
    (out, widths, wm)
}

/// Line indices in batches of similar width (narrowest first).
pub fn width_batches(widths: &[usize], max_batch: usize) -> Vec<Vec<usize>> {
    let (max_ratio, min_width, max_area) = (1.3f64, 64usize, 64usize * 640);
    let mut order: Vec<usize> = (0..widths.len()).collect();
    order.sort_by_key(|&i| widths[i]); // stable, as Python's sorted
    let mut batches: Vec<Vec<usize>> = vec![];
    for i in order {
        let fits = batches.last().map(|b: &Vec<usize>| {
            b.len() < max_batch
                && widths[i].max(min_width) as f64 <= max_ratio * widths[b[0]].max(min_width) as f64
                && (b.len() + 1) * widths[i].max(min_width) <= max_area
        });
        if fits == Some(true) {
            batches.last_mut().unwrap().push(i);
        } else {
            batches.push(vec![i]);
        }
    }
    batches
}
