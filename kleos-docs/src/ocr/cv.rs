//! The few image operations the OCR pipeline needs, written to match OpenCV's
//! results (the Python package uses `cv2`): area and linear resizing of 8-bit
//! images, affine rotation, connected components and quarter turns.
//!
//! Linear resizing follows OpenCV's fixed-point arithmetic for `CV_8U`
//! (11-bit coefficients, rounding on the 22-bit product), so most pixels come
//! out identical; area resizing uses OpenCV's overlap weights.  Rotation is
//! bilinear in floating point (OpenCV quantizes coordinates to 1/32 pixel):
//! it feeds the skew search and the deskewed page, where a grey level of
//! difference does not change a reading.

/// An 8-bit grayscale image, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Gray {
    pub fn new(w: usize, h: usize, fill: u8) -> Self {
        Gray { w, h, data: vec![fill; w * h] }
    }
    pub fn from_raw(w: usize, h: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), w * h);
        Gray { w, h, data }
    }
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }
    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.w..(y + 1) * self.w]
    }
    /// The sub-image [x0, x1) x [y0, y1) (clipped to the image).
    pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Gray {
        let x1 = x1.min(self.w);
        let y1 = y1.min(self.h);
        let x0 = x0.min(x1);
        let y0 = y0.min(y1);
        let mut out = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for y in y0..y1 {
            out.extend_from_slice(&self.data[y * self.w + x0..y * self.w + x1]);
        }
        Gray { w: x1 - x0, h: y1 - y0, data: out }
    }
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().map(|&v| v as u64).sum::<u64>() as f64 / self.data.len() as f64
    }
    /// 255 - v for every pixel.
    pub fn inverted(&self) -> Gray {
        Gray { w: self.w, h: self.h, data: self.data.iter().map(|v| 255 - v).collect() }
    }
}

/// OpenCV's `cvRound` (round half to even, as `lrint`).
#[inline]
pub fn cv_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

#[inline]
fn sat_u8(v: i64) -> u8 {
    v.clamp(0, 255) as u8
}

/// `cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)` for 8-bit images.
/// Upscaling falls back to linear, as OpenCV does.
pub fn resize_area(src: &Gray, w: usize, h: usize) -> Gray {
    if w == src.w && h == src.h {
        return src.clone();
    }
    if w > src.w || h > src.h {
        // OpenCV: INTER_AREA is area averaging only when neither axis grows;
        // otherwise it is INTER_LINEAR
        return resize_linear(src, w, h);
    }
    let sx = src.w as f64 / w as f64;
    let sy = src.h as f64 / h as f64;
    let ix = sx.round();
    let iy = sy.round();
    if (sx - ix).abs() < f64::EPSILON && (sy - iy).abs() < f64::EPSILON && ix >= 1.0 && iy >= 1.0 {
        return resize_area_fast(src, w, h, ix as usize, iy as usize);
    }
    let xt = area_tab(src.w, w, sx);
    let yt = area_tab(src.h, h, sy);
    // horizontal pass of one source row at a time into a reused buffer,
    // then its vertical contributions. yt is ordered by (dy, syi), so every
    // destination row still receives its terms in the same order as a pass
    // over all rows would give it (bit-identical results, one row of memory)
    let mut by_src: Vec<Vec<(usize, f64)>> = vec![vec![]; src.h];
    for &(dy, syi, b) in &yt {
        by_src[syi].push((dy, b));
    }
    let mut row_buf = vec![0f64; w];
    let mut acc = vec![0f64; w * h];
    for (y, terms) in by_src.iter().enumerate() {
        if terms.is_empty() {
            continue;
        }
        row_buf.iter_mut().for_each(|v| *v = 0.0);
        let r = src.row(y);
        for &(dx, sxi, a) in &xt {
            row_buf[dx] += r[sxi] as f64 * a;
        }
        for &(dy, b) in terms {
            let o = &mut acc[dy * w..(dy + 1) * w];
            for x in 0..w {
                o[x] += row_buf[x] * b;
            }
        }
    }
    Gray { w, h, data: acc.iter().map(|&v| sat_u8(cv_round(v))).collect() }
}

/// OpenCV's `computeResizeAreaTab`: (dst index, src index, weight), the
/// weights of one destination pixel summing to 1.
fn area_tab(ssize: usize, dsize: usize, scale: f64) -> Vec<(usize, usize, f64)> {
    let mut tab = Vec::new();
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let sx2 = (fsx2.floor() as i64).min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if (sx1 as f64 - fsx1) > 1e-3 {
            tab.push((dx, (sx1 - 1) as usize, (sx1 as f64 - fsx1) / cell));
        }
        for sx in sx1..sx2 {
            tab.push((dx, sx as usize, 1.0 / cell));
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push((dx, sx2 as usize, ((fsx2 - sx2 as f64).min(1.0).min(cell)) / cell));
        }
    }
    tab
}

fn resize_area_fast(src: &Gray, w: usize, h: usize, fx: usize, fy: usize) -> Gray {
    let area = (fx * fy) as u32;
    let mut out = vec![0u8; w * h];
    for dy in 0..h {
        for dx in 0..w {
            let mut s = 0u32;
            for yy in 0..fy {
                let r = &src.data[(dy * fy + yy) * src.w + dx * fx..];
                for xx in 0..fx {
                    s += r[xx] as u32;
                }
            }
            out[dy * w + dx] = ((s + area / 2) / area) as u8;
        }
    }
    Gray { w, h, data: out }
}

const COEF_BITS: i32 = 11;
const COEF_SCALE: i32 = 1 << COEF_BITS;

/// Per destination index: (source index, its fixed-point weight, the next
/// source index, its weight), OpenCV's INTER_LINEAR table.
fn linear_tab(ssize: usize, dsize: usize) -> Vec<(usize, i32, usize, i32)> {
    let scale = ssize as f64 / dsize as f64;
    (0..dsize)
        .map(|d| {
            let mut f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let mut s = f.floor() as i64;
            f -= s as f32;
            if s < 0 {
                s = 0;
                f = 0.0;
            }
            if s >= ssize as i64 - 1 {
                s = ssize as i64 - 1;
                f = 0.0;
            }
            let c0 = ((1.0 - f) * COEF_SCALE as f32).round_ties_even() as i32;
            let c0 = c0.clamp(i16::MIN as i32, i16::MAX as i32);
            let s = s as usize;
            (s, c0, (s + 1).min(ssize - 1), COEF_SCALE - c0)
        })
        .collect()
}

/// `cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)` for 8-bit images
/// (OpenCV's fixed-point arithmetic).
pub fn resize_linear(src: &Gray, w: usize, h: usize) -> Gray {
    if w == src.w && h == src.h {
        return src.clone();
    }
    let xt = linear_tab(src.w, w);
    let yt = linear_tab(src.h, h);
    let mut cache: Vec<Option<Vec<i32>>> = vec![None; src.h];
    let hpass = |y: usize| -> Vec<i32> {
        let r = src.row(y);
        xt.iter().map(|&(a, ca, b, cb)| r[a] as i32 * ca + r[b] as i32 * cb).collect()
    };
    let mut out = vec![0u8; w * h];
    for (dy, &(y0, b0, y1, b1)) in yt.iter().enumerate() {
        if cache[y0].is_none() {
            cache[y0] = Some(hpass(y0));
        }
        if cache[y1].is_none() {
            cache[y1] = Some(hpass(y1));
        }
        let r0 = cache[y0].as_ref().unwrap();
        let r1 = cache[y1].as_ref().unwrap();
        let o = &mut out[dy * w..(dy + 1) * w];
        for x in 0..w {
            // VResizeLinear with FixedPtCast<int, uchar, 22>: the 16-bit
            // beta times the row values, rounded on the 22-bit shift
            let v = ((b0 as i64 * r0[x] as i64) + (b1 as i64 * r1[x] as i64) + (1 << 21)) >> 22;
            o[x] = sat_u8(v);
        }
    }
    Gray { w, h, data: out }
}

/// Linear resize of a float map (bilinear, OpenCV's half-pixel centres).
pub fn resize_linear_f32(src: &[f32], sw: usize, sh: usize, w: usize, h: usize) -> Vec<f32> {
    let sxs = sw as f64 / w as f64;
    let sys = sh as f64 / h as f64;
    let tab = |ssize: usize, d: usize, scale: f64| {
        let mut f = (d as f64 + 0.5) * scale - 0.5;
        let mut s = f.floor() as i64;
        f -= s as f64;
        if s < 0 {
            s = 0;
            f = 0.0;
        }
        if s >= ssize as i64 - 1 {
            s = ssize as i64 - 1;
            f = 0.0;
        }
        (s as usize, f as f32)
    };
    let mut out = vec![0f32; w * h];
    for dy in 0..h {
        let (y0, fy) = tab(sh, dy, sys);
        let y1 = (y0 + 1).min(sh - 1);
        for dx in 0..w {
            let (x0, fx) = tab(sw, dx, sxs);
            let x1 = (x0 + 1).min(sw - 1);
            let a = src[y0 * sw + x0] * (1.0 - fx) + src[y0 * sw + x1] * fx;
            let b = src[y1 * sw + x0] * (1.0 - fx) + src[y1 * sw + x1] * fx;
            out[dy * w + dx] = a * (1.0 - fy) + b * fy;
        }
    }
    out
}

/// `cv2.getRotationMatrix2D(center, angle, 1.0)`: [2, 3].
pub fn rotation_matrix(cx: f64, cy: f64, degrees: f64) -> [[f64; 3]; 2] {
    let a = degrees.to_radians();
    let (al, be) = (a.cos(), a.sin());
    [[al, be, (1.0 - al) * cx - be * cy], [-be, al, be * cx + (1.0 - al) * cy]]
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Border {
    /// pixels outside take this value
    Constant(u8),
    /// the nearest edge pixel
    Replicate,
}

/// `cv2.warpAffine(src, M, (w, h))` with linear interpolation: M maps source
/// to destination; each destination pixel samples the inverse map.
pub fn warp_affine(src: &Gray, m: [[f64; 3]; 2], w: usize, h: usize, border: Border) -> Gray {
    // invert the 2x3 affine
    let d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let a11 = m[1][1] * d;
    let a22 = m[0][0] * d;
    let a12 = -m[0][1] * d;
    let a21 = -m[1][0] * d;
    let b1 = -a11 * m[0][2] - a12 * m[1][2];
    let b2 = -a21 * m[0][2] - a22 * m[1][2];
    let mut out = vec![0u8; w * h];
    let sw = src.w as i64;
    let sh = src.h as i64;
    let get = |x: i64, y: i64| -> f32 {
        if x >= 0 && y >= 0 && x < sw && y < sh {
            src.data[(y * sw + x) as usize] as f32
        } else {
            match border {
                Border::Constant(c) => c as f32,
                Border::Replicate => src.data[(y.clamp(0, sh - 1) * sw + x.clamp(0, sw - 1)) as usize] as f32,
            }
        }
    };
    for y in 0..h {
        for x in 0..w {
            let sx = a11 * x as f64 + a12 * y as f64 + b1;
            let sy = a21 * x as f64 + a22 * y as f64 + b2;
            // OpenCV quantizes to 1/32 pixel (INTER_TAB_SIZE)
            let qx = (sx * 32.0).round() / 32.0;
            let qy = (sy * 32.0).round() / 32.0;
            let x0 = qx.floor() as i64;
            let y0 = qy.floor() as i64;
            let fx = (qx - x0 as f64) as f32;
            let fy = (qy - y0 as f64) as f32;
            let v = (get(x0, y0) * (1.0 - fx) + get(x0 + 1, y0) * fx) * (1.0 - fy)
                + (get(x0, y0 + 1) * (1.0 - fx) + get(x0 + 1, y0 + 1) * fx) * fy;
            out[y * w + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    Gray { w, h, data: out }
}

/// Per destination row, the sum of `warp_affine(src, m, w, h, Constant(0))`
/// (rounded per pixel, as the image would hold), computed without the image.
pub fn warp_affine_row_sums(src: &Gray, m: [[f64; 3]; 2], w: usize, h: usize) -> Vec<f64> {
    let d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a22, a12, a21) = (m[1][1] * d, m[0][0] * d, -m[0][1] * d, -m[1][0] * d);
    let b1 = -a11 * m[0][2] - a12 * m[1][2];
    let b2 = -a21 * m[0][2] - a22 * m[1][2];
    let (sw, sh) = (src.w as i64, src.h as i64);
    let data = &src.data;
    let get = |x: i64, y: i64| -> f32 {
        if x >= 0 && y >= 0 && x < sw && y < sh {
            data[(y * sw + x) as usize] as f32
        } else {
            0.0
        }
    };
    // The sums are integers, so skipping terms that are exactly zero leaves
    // them unchanged. A destination pixel can only be non-zero when its 2x2
    // source neighbourhood touches the bounding box of the non-zero source
    // pixels: per destination row, only the x range whose source point falls
    // within that box (plus a 2-pixel margin for rounding) is visited, and a
    // neighbourhood of four zeros is skipped without interpolating.
    let mut out = vec![0f64; h];
    let (mut bx0, mut by0, mut bx1, mut by1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    // inked source rows before each row (blank rows between text lines let a
    // whole destination row be skipped: its sum is exactly 0)
    let mut inked_before = vec![0u32; sh as usize + 1];
    for yy in 0..sh {
        let row = &data[(yy * sw) as usize..((yy + 1) * sw) as usize];
        let inked = row.iter().any(|&v| v != 0);
        inked_before[yy as usize + 1] = inked_before[yy as usize] + inked as u32;
        if !inked {
            continue;
        }
        if let Some(f) = row.iter().position(|&v| v != 0) {
            let l = row.iter().rposition(|&v| v != 0).unwrap();
            bx0 = bx0.min(f as i64);
            bx1 = bx1.max(l as i64);
            by0 = by0.min(yy);
            by1 = by1.max(yy);
        }
    }
    if bx0 > bx1 {
        return out;
    }
    let (lx, hx, ly, hy) = ((bx0 - 3) as f64, (bx1 + 3) as f64, (by0 - 3) as f64, (by1 + 3) as f64);
    // the x where a linear function c0 + c1 * x lies in [lo, hi]
    let span = |c0: f64, c1: f64, lo: f64, hi: f64| -> (f64, f64) {
        if c1.abs() < 1e-12 {
            if c0 >= lo && c0 <= hi { (f64::NEG_INFINITY, f64::INFINITY) } else { (1.0, 0.0) }
        } else {
            let (p, q) = ((lo - c0) / c1, (hi - c0) / c1);
            (p.min(q), p.max(q))
        }
    };
    let (mut ixs, mut iys): (Vec<i64>, Vec<i64>) = (Vec::with_capacity(w), Vec::with_capacity(w));
    for (y, o) in out.iter_mut().enumerate() {
        let (ax0, ax1) = span(a12 * y as f64 + b1, a11, lx, hx);
        let (ay0, ay1) = span(a22 * y as f64 + b2, a21, ly, hy);
        let (xa, xb) = (ax0.max(ay0), ax1.min(ay1));
        if xa > xb {
            continue;
        }
        let xs = (xa.floor() as i64 - 1).clamp(0, w as i64) as usize;
        let xe = (xb.ceil() as i64 + 2).clamp(0, w as i64) as usize;
        let mut acc = 0u64;
        // the same sums as `a11 * x + a12 * y + b1` (left to right), with the
        // y terms hoisted; the 1/32 quantisation in integers: round(s * 32)
        // gives floor (>> 5) and fraction (& 31) / 32 exactly. The source
        // positions of the row are computed first (a loop the compiler
        // vectorises), then the pixels are read.
        let (cy1, cy2) = (a12 * y as f64, a22 * y as f64);
        // the source rows this destination row reads (y is linear in x, so
        // its ends bound it): none inked, nothing to add
        let row_of = |x: usize| (((a21 * x as f64 + cy2 + b2) * 32.0).round() as i64) >> 5;
        let (ya, yb) = (row_of(xs), row_of(xe.saturating_sub(1).max(xs)));
        let (r0, r1) = ((ya.min(yb)).clamp(0, sh), (ya.max(yb) + 2).clamp(0, sh));
        if inked_before[r1 as usize] == inked_before[r0 as usize] {
            continue;
        }
        ixs.clear();
        iys.clear();
        ixs.extend((xs..xe).map(|x| ((a11 * x as f64 + cy1 + b1) * 32.0).round() as i64));
        iys.extend((xs..xe).map(|x| ((a21 * x as f64 + cy2 + b2) * 32.0).round() as i64));
        for (&ix, &iy) in ixs.iter().zip(iys.iter()) {
            let x0 = ix >> 5;
            let y0 = iy >> 5;
            // whole 2x2 neighbourhood outside: zero
            if x0 >= -1 && y0 >= -1 && x0 < sw && y0 < sh {
                let (p00, p10, p01, p11) = if x0 >= 0 && y0 >= 0 && x0 + 1 < sw && y0 + 1 < sh {
                    // inside: direct reads; a neighbourhood of four zeros
                    // (most of an ink image) interpolates to exactly 0
                    let i = (y0 * sw + x0) as usize;
                    let j = i + sw as usize;
                    // SAFETY: 0 <= x0 < sw - 1 and 0 <= y0 < sh - 1, so i + 1
                    // and j + 1 are inside the sw * sh buffer
                    let (q00, q10, q01, q11) = unsafe { (*data.get_unchecked(i), *data.get_unchecked(i + 1), *data.get_unchecked(j), *data.get_unchecked(j + 1)) };
                    if (q00 | q10 | q01 | q11) == 0 {
                        continue;
                    }
                    (q00 as f32, q10 as f32, q01 as f32, q11 as f32)
                } else {
                    (get(x0, y0), get(x0 + 1, y0), get(x0, y0 + 1), get(x0 + 1, y0 + 1))
                };
                let fx = (ix & 31) as f32 / 32.0;
                let fy = (iy & 31) as f32 / 32.0;
                let v = (p00 * (1.0 - fx) + p10 * fx) * (1.0 - fy) + (p01 * (1.0 - fx) + p11 * fx) * fy;
                acc += v.round().clamp(0.0, 255.0) as u64;
            }
        }
        *o = acc as f64;
    }
    out
}

/// `np.rot90(a, k)`: k quarter turns counter-clockwise (negative: clockwise).
pub fn rot90(src: &Gray, k: i32) -> Gray {
    let k = k.rem_euclid(4);
    match k {
        0 => src.clone(),
        2 => {
            let mut d = src.data.clone();
            d.reverse();
            Gray { w: src.w, h: src.h, data: d }
        }
        1 => {
            // counter-clockwise: out[y][x] = src[x][w-1-y], out is (h=src.w, w=src.h)
            let (w, h) = (src.h, src.w);
            let mut d = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    d[y * w + x] = src.data[x * src.w + (src.w - 1 - y)];
                }
            }
            Gray { w, h, data: d }
        }
        _ => {
            // clockwise: out[y][x] = src[h-1-x][y]
            let (w, h) = (src.h, src.w);
            let mut d = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    d[y * w + x] = src.data[(src.h - 1 - x) * src.w + y];
                }
            }
            Gray { w, h, data: d }
        }
    }
}

/// One component of `connected_components`: bounding box and area.
#[derive(Clone, Copy, Debug)]
pub struct CompStat {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
    pub area: usize,
}

/// `cv2.connectedComponentsWithStats(mask, connectivity)`: labels (0 =
/// background, components numbered in raster order of their first pixel) and
/// stats per label (index 0 is the background, unused).
pub fn connected_components(mask: &[bool], w: usize, h: usize, eight: bool) -> (Vec<u32>, Vec<CompStat>) {
    let mut labels = vec![0u32; w * h];
    let mut parent: Vec<u32> = vec![0];
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    fn union(p: &mut [u32], a: u32, b: u32) -> u32 {
        let ra = find(p, a);
        let rb = find(p, b);
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        p[hi as usize] = lo;
        lo
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !mask[i] {
                continue;
            }
            let mut l = 0u32;
            let take = |n: u32, l: &mut u32, p: &mut Vec<u32>| {
                if n != 0 {
                    *l = if *l == 0 { n } else { union(p, *l, n) };
                }
            };
            if x > 0 {
                take(labels[i - 1], &mut l, &mut parent);
            }
            if y > 0 {
                take(labels[i - w], &mut l, &mut parent);
                if eight {
                    if x > 0 {
                        take(labels[i - w - 1], &mut l, &mut parent);
                    }
                    if x + 1 < w {
                        take(labels[i - w + 1], &mut l, &mut parent);
                    }
                }
            }
            if l == 0 {
                l = parent.len() as u32;
                parent.push(l);
            }
            labels[i] = l;
        }
    }
    // relabel roots in order of first appearance (raster order)
    let mut map = vec![0u32; parent.len()];
    let mut stats = vec![CompStat { left: 0, top: 0, width: 0, height: 0, area: 0 }];
    let mut bounds: Vec<(usize, usize, usize, usize)> = vec![(0, 0, 0, 0)];
    let mut next = 1u32;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let l = labels[i];
            if l == 0 {
                continue;
            }
            let r = find(&mut parent, l);
            if map[r as usize] == 0 {
                map[r as usize] = next;
                next += 1;
                stats.push(CompStat { left: x, top: y, width: 0, height: 0, area: 0 });
                bounds.push((x, y, x, y));
            }
            let nl = map[r as usize];
            labels[i] = nl;
            let b = &mut bounds[nl as usize];
            b.0 = b.0.min(x);
            b.1 = b.1.min(y);
            b.2 = b.2.max(x);
            b.3 = b.3.max(y);
            stats[nl as usize].area += 1;
        }
    }
    for (k, b) in bounds.iter().enumerate().skip(1) {
        let s = &mut stats[k];
        s.left = b.0;
        s.top = b.1;
        s.width = b.2 - b.0 + 1;
        s.height = b.3 - b.1 + 1;
    }
    (labels, stats)
}

/// RGB to gray with OpenCV's weights (fixed point, `cv2.COLOR_RGB2GRAY`).
pub fn rgb_to_gray(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 4899 + g as u32 * 9617 + b as u32 * 1868 + 8192) >> 14) as u8
}

/// np.percentile (linear) of an 8-bit image at q in [0, 1], from its histogram.
pub fn percentile_u8(hist: &[u64; 256], n: u64, q: f64) -> f64 {
    let mut cum = [0u64; 256];
    let mut s = 0;
    for i in 0..256 {
        s += hist[i];
        cum[i] = s;
    }
    let x = q * (n.saturating_sub(1)) as f64;
    let i = x.floor() as u64;
    let g = x - i as f64;
    let ss = |t: u64| cum.iter().position(|&c| c > t).unwrap_or(255) as f64;
    let lo = ss(i);
    let hi = ss((i + 1).min(n.saturating_sub(1)));
    let d = hi - lo;
    if g >= 0.5 {
        hi - d * (1.0 - g)
    } else {
        lo + d * g
    }
}

#[cfg(test)]
mod tests {
    // the original general-path area resize, for the equality check below
    fn resize_area_ref(src: &Gray, w: usize, h: usize) -> Gray {
        if w == src.w && h == src.h {
            return src.clone();
        }
        if w > src.w || h > src.h {
            // OpenCV: INTER_AREA is area averaging only when neither axis grows;
            // otherwise it is INTER_LINEAR
            return resize_linear(src, w, h);
        }
        let sx = src.w as f64 / w as f64;
        let sy = src.h as f64 / h as f64;
        let ix = sx.round();
        let iy = sy.round();
        if (sx - ix).abs() < f64::EPSILON && (sy - iy).abs() < f64::EPSILON && ix >= 1.0 && iy >= 1.0 {
            return resize_area_fast(src, w, h, ix as usize, iy as usize);
        }
        let xt = area_tab(src.w, w, sx);
        let yt = area_tab(src.h, h, sy);
        // horizontal pass into f64 rows, then vertical accumulate
        let mut hrow = vec![vec![0f64; w]; src.h];
        for y in 0..src.h {
            let r = src.row(y);
            let out = &mut hrow[y];
            for &(dx, sxi, a) in &xt {
                out[dx] += r[sxi] as f64 * a;
            }
        }
        let mut acc = vec![0f64; w * h];
        for &(dy, syi, b) in &yt {
            let row = &hrow[syi];
            let o = &mut acc[dy * w..(dy + 1) * w];
            for x in 0..w {
                o[x] += row[x] * b;
            }
        }
        Gray { w, h, data: acc.iter().map(|&v| sat_u8(cv_round(v))).collect() }
    }
    

    #[test]
    fn resize_area_matches_the_reference_exactly() {
        let mut seed = 3u64;
        let data: Vec<u8> = (0..928 * 1472)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (seed >> 56) as u8
            })
            .collect();
        let g = Gray { w: 928, h: 1472, data };
        for (w, h) in [(504, 800), (252, 400), (768, 1218), (300, 333)] {
            let t = std::time::Instant::now();
            let a = resize_area(&g, w, h);
            let t_new = t.elapsed();
            let t = std::time::Instant::now();
            let b = resize_area_ref(&g, w, h);
            let t_ref = t.elapsed();
            assert_eq!(a.data, b.data, "{w}x{h}");
            eprintln!("resize_area {w}x{h}: new {t_new:?} ref {t_ref:?}");
        }
    }

    // the original kernel, for the equality check below
    fn row_sums_ref(src: &Gray, m: [[f64; 3]; 2], w: usize, h: usize) -> Vec<f64> {
        let d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let d = if d != 0.0 { 1.0 / d } else { 0.0 };
        let (a11, a22, a12, a21) = (m[1][1] * d, m[0][0] * d, -m[0][1] * d, -m[1][0] * d);
        let b1 = -a11 * m[0][2] - a12 * m[1][2];
        let b2 = -a21 * m[0][2] - a22 * m[1][2];
        let (sw, sh) = (src.w as i64, src.h as i64);
        let data = &src.data;
        let get = |x: i64, y: i64| -> f32 {
            if x >= 0 && y >= 0 && x < sw && y < sh {
                data[(y * sw + x) as usize] as f32
            } else {
                0.0
            }
        };
        let mut out = vec![0f64; h];
        for (y, o) in out.iter_mut().enumerate() {
            let mut acc = 0u64;
            for x in 0..w {
                let sx = a11 * x as f64 + a12 * y as f64 + b1;
                let sy = a21 * x as f64 + a22 * y as f64 + b2;
                let qx = (sx * 32.0).round() / 32.0;
                let qy = (sy * 32.0).round() / 32.0;
                let x0 = qx.floor() as i64;
                let y0 = qy.floor() as i64;
                // whole 2x2 neighbourhood outside: zero
                if x0 >= -1 && y0 >= -1 && x0 < sw && y0 < sh {
                    let fx = (qx - x0 as f64) as f32;
                    let fy = (qy - y0 as f64) as f32;
                    let v = (get(x0, y0) * (1.0 - fx) + get(x0 + 1, y0) * fx) * (1.0 - fy) + (get(x0, y0 + 1) * (1.0 - fx) + get(x0 + 1, y0 + 1) * fx) * fy;
                    acc += v.round().clamp(0.0, 255.0) as u64;
                }
            }
            *o = acc as f64;
        }
        out
    }
    

    #[test]
    fn row_sums_match_the_reference_exactly() {
        let (w, h) = (300usize, 420usize);
        let mut seed = 7u64;
        let mut data = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let r = (seed >> 33) as u32;
                // text-like rows of ink inside a margin, a little noise elsewhere
                let ink = x > 30 && x < 270 && y > 40 && y < 380 && (y / 9) % 2 == 0 && r % 3 == 0;
                data[y * w + x] = if ink { 120 + (r % 130) as u8 } else if r % 50 == 0 { (r % 9) as u8 } else { 0 };
            }
        }
        let g = Gray { w, h, data };
        for k in 0..60 {
            let a = -45.0 + k as f64 * 1.53;
            let m = rotation_matrix(w as f64 / 2.0, h as f64 / 2.0, a);
            assert_eq!(warp_affine_row_sums(&g, m, w, h), row_sums_ref(&g, m, w, h), "angle {a}");
        }
        // timing on a page-sized ink map (cargo test --release -- --nocapture)
        let (w2, h2) = (800usize, 1270usize);
        let mut d2 = vec![0u8; w2 * h2];
        for (i, v) in d2.iter_mut().enumerate() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let (x, y) = (i % w2, i / w2);
            let r = (seed >> 33) as u32;
            *v = if x > 60 && x < 740 && y > 80 && y < 1190 && (y / 14) % 2 == 0 && r % 3 == 0 { 150 } else if r % 4 == 0 { 3 } else { 0 };
        }
        let g2 = Gray { w: w2, h: h2, data: d2 };
        let ms: Vec<_> = (0..24).map(|k| rotation_matrix(400.0, 635.0, -6.0 + k as f64 * 0.5)).collect();
        let t = std::time::Instant::now();
        let a: Vec<_> = ms.iter().map(|m| warp_affine_row_sums(&g2, *m, w2, h2)).collect();
        let t_new = t.elapsed();
        let t = std::time::Instant::now();
        let b: Vec<_> = ms.iter().map(|m| row_sums_ref(&g2, *m, w2, h2)).collect();
        let t_ref = t.elapsed();
        assert_eq!(a, b);
        eprintln!("row sums, 24 angles 800x1270: new {t_new:?} ref {t_ref:?}");
        let blank = Gray { w: 50, h: 40, data: vec![0; 2000] };
        let m = rotation_matrix(25.0, 20.0, 7.0);
        assert_eq!(warp_affine_row_sums(&blank, m, 50, 40), row_sums_ref(&blank, m, 50, 40));
    }

    use super::*;

    #[test]
    fn rot90_round_trips() {
        let g = Gray::from_raw(3, 2, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(rot90(&rot90(&g, 1), -1), g);
        assert_eq!(rot90(&g, 1).data, vec![3, 6, 2, 5, 1, 4]);
        assert_eq!(rot90(&g, -1).data, vec![4, 1, 5, 2, 6, 3]);
        assert_eq!(rot90(&rot90(&g, 2), 2), g);
    }

    #[test]
    fn components_raster_order() {
        #[rustfmt::skip]
        let m = [
            false, true, false, true,
            false, true, false, false,
            true, false, false, true,
        ];
        let (l, s) = connected_components(&m, 4, 3, false);
        assert_eq!(s.len(), 5);
        assert_eq!(l[1], 1);
        assert_eq!(l[3], 2);
        assert_eq!(l[8], 3);
        assert_eq!(s[1].height, 2);
        let (_, s8) = connected_components(&m, 4, 3, true);
        assert_eq!(s8.len(), 4); // (2,0) joins column 1 diagonally
    }

    #[test]
    fn resize_identity_and_halves() {
        let g = Gray::from_raw(4, 2, vec![0, 255, 0, 255, 100, 100, 200, 200]);
        let a = resize_area(&g, 2, 1);
        assert_eq!(a.data, vec![114, 164]);
        let l = resize_linear(&g, 8, 4);
        assert_eq!(l.w, 8);
        assert_eq!(l.at(0, 0), 0);
    }
}
