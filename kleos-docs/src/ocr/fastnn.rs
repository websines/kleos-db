//! A hand-written CPU path for Indra's small CNNs (the recognizers first).
//!
//! tract runs these graphs correctly but spends most of its time moving axes
//! and building im2col buffers around tiny convolutions (see
//! docs/specs/documents.md, "Starting point for a faster CPU path"). Here the
//! recognizer is written out directly in a channels-last layout
//! (`[batch][time][channel]`):
//! - every 1×1 convolution, the stems and the classifier are one GEMM
//!   (Apple Accelerate, i.e. the AMX units, on macOS; `matrixmultiply`
//!   elsewhere);
//! - depthwise convolutions are multiply-adds over contiguous channels, which
//!   the compiler vectorizes;
//! - HardSwish, the masked squeeze-excite, the residual and the width mask
//!   are fused into the passes around the GEMMs.
//!
//! The weights are read from the same ONNX files by node name, so the graph
//! stays the source of truth; a graph that does not have the expected shape is
//! refused at load and the caller keeps tract. tract also stays available as
//! the reference (`KLEOS_OCR_TRACT=1`).

use std::collections::HashMap;
use std::path::Path;

use tract_onnx::pb::{GraphProto, NodeProto, TensorProto};
use tract_onnx::prelude::Framework;

use super::models::{ModelError, Output};

/// How many top-k candidates per step the recognizers export for the LM.
pub const TOPK: usize = 8;

// ---------------------------------------------------------------- kernels

#[cfg(target_os = "macos")]
mod blas {
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_sgemm(
            order: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
        fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
    }
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    /// `c[m×n] (+)= a[m×k] · w[n×k]ᵀ`
    pub fn gemm_nt(m: usize, n: usize, k: usize, a: &[f32], w: &[f32], c: &mut [f32], accumulate: bool) {
        debug_assert!(a.len() >= m * k && w.len() >= n * k && c.len() >= m * n);
        if m == 0 || n == 0 {
            return;
        }
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                w.as_ptr(),
                k as i32,
                if accumulate { 1.0 } else { 0.0 },
                c.as_mut_ptr(),
                n as i32,
            )
        }
    }

    /// `c[m×n] (+)= a[m×k] · b[k×n]`
    pub fn gemm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32], accumulate: bool) {
        if m == 0 || n == 0 {
            return;
        }
        unsafe {
            cblas_sgemm(ROW_MAJOR, NO_TRANS, NO_TRANS, m as i32, n as i32, k as i32, 1.0, a.as_ptr(), k as i32, b.as_ptr(), n as i32, if accumulate { 1.0 } else { 0.0 }, c.as_mut_ptr(), n as i32)
        }
    }

    /// In-place `x = exp(x)`.
    pub fn exp_inplace(x: &mut [f32]) {
        let n = x.len() as i32;
        if n > 0 {
            unsafe { vvexpf(x.as_mut_ptr(), x.as_ptr(), &n) }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod blas {
    /// `c[m×n] (+)= a[m×k] · w[n×k]ᵀ`
    pub fn gemm_nt(m: usize, n: usize, k: usize, a: &[f32], w: &[f32], c: &mut [f32], accumulate: bool) {
        if m == 0 || n == 0 {
            return;
        }
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                w.as_ptr(),
                1,
                k as isize,
                if accumulate { 1.0 } else { 0.0 },
                c.as_mut_ptr(),
                n as isize,
                1,
            )
        }
    }

    /// `c[m×n] (+)= a[m×k] · b[k×n]`
    pub fn gemm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32], accumulate: bool) {
        if m == 0 || n == 0 {
            return;
        }
        unsafe {
            matrixmultiply::sgemm(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1, if accumulate { 1.0 } else { 0.0 }, c.as_mut_ptr(), n as isize, 1)
        }
    }

    pub fn exp_inplace(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = v.exp();
        }
    }
}

/// `c[m×n] (+)= a[m×k] · b[k×n]` (for the 2-D nets).
pub fn blas_gemm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32], accumulate: bool) {
    blas::gemm_nn(m, n, k, a, b, c, accumulate)
}

/// `out[m×n] = a[m×k] · w[n×k]ᵀ + bias[n]`
fn linear(m: usize, n: usize, k: usize, a: &[f32], w: &[f32], bias: &[f32], out: &mut Vec<f32>) {
    out.clear();
    out.reserve(m * n);
    for _ in 0..m {
        out.extend_from_slice(bias);
    }
    blas::gemm_nt(m, n, k, a, w, out, true);
}

#[inline(always)]
fn hardswish(x: f32) -> f32 {
    x * (x + 3.0).clamp(0.0, 6.0) * (1.0 / 6.0)
}

fn hardswish_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = hardswish(*v);
    }
}

/// Max of a slice (8 lanes, so it vectorizes).
fn max_f32(x: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 8];
    let mut it = x.chunks_exact(8);
    for c in &mut it {
        for i in 0..8 {
            acc[i] = if c[i] > acc[i] { c[i] } else { acc[i] };
        }
    }
    let mut m = acc.iter().copied().fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    for &v in it.remainder() {
        if v > m {
            m = v;
        }
    }
    m
}

/// Sum of a slice (8 lanes).
fn sum_f32(x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut it = x.chunks_exact(8);
    for c in &mut it {
        for i in 0..8 {
            acc[i] += c[i];
        }
    }
    acc.iter().sum::<f32>() + it.remainder().iter().sum::<f32>()
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------- weights

/// The parts of an ONNX graph the hand-written nets read.
struct Graph {
    nodes: HashMap<String, NodeProto>,
    tensors: HashMap<String, TensorProto>,
}

fn proto_f32(t: &TensorProto) -> Option<Vec<f32>> {
    match t.data_type {
        1 => {
            if !t.raw_data.is_empty() {
                Some(t.raw_data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
            } else {
                Some(t.float_data.clone())
            }
        }
        _ => None,
    }
}

fn proto_i64(t: &TensorProto) -> Option<Vec<i64>> {
    match t.data_type {
        7 => {
            if !t.raw_data.is_empty() {
                Some(t.raw_data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
            } else {
                Some(t.int64_data.clone())
            }
        }
        _ => None,
    }
}

impl Graph {
    fn load(path: &Path) -> Result<Self, String> {
        let proto = tract_onnx::onnx().proto_model_for_path(path).map_err(|e| format!("{e:?}"))?;
        let g: GraphProto = proto.graph.ok_or("no graph")?;
        let mut tensors: HashMap<String, TensorProto> = g.initializer.into_iter().map(|t| (t.name.clone(), t)).collect();
        let mut nodes = HashMap::new();
        for n in g.node {
            if n.op_type == "Constant" {
                if let (Some(out), Some(t)) = (n.output.first(), n.attribute.iter().find(|a| a.name == "value").and_then(|a| a.t.clone())) {
                    tensors.insert(out.clone(), t);
                }
                continue;
            }
            nodes.insert(n.name.clone(), n);
        }
        Ok(Graph { nodes, tensors })
    }

    fn node(&self, name: &str) -> Result<&NodeProto, String> {
        self.nodes.get(name).ok_or_else(|| format!("no node {name}"))
    }

    fn has(&self, name: &str) -> bool {
        self.nodes.contains_key(name)
    }

    /// Input `i` of node `name` as f32 data of the expected element count.
    fn f32_input(&self, name: &str, i: usize, len: usize) -> Result<Vec<f32>, String> {
        let n = self.node(name)?;
        let key = n.input.get(i).ok_or_else(|| format!("{name}: no input {i}"))?;
        let t = self.tensors.get(key).ok_or_else(|| format!("{name}: input {key} is not constant"))?;
        let v = proto_f32(t).ok_or_else(|| format!("{name}: input {key} is not f32"))?;
        if v.len() != len {
            return Err(format!("{name}: input {key} has {} values, expected {len}", v.len()));
        }
        Ok(v)
    }

    fn i64_input(&self, name: &str, i: usize) -> Result<Vec<i64>, String> {
        let n = self.node(name)?;
        let key = n.input.get(i).ok_or_else(|| format!("{name}: no input {i}"))?;
        let t = self.tensors.get(key).ok_or_else(|| format!("{name}: input {key} is not constant"))?;
        proto_i64(t).ok_or_else(|| format!("{name}: input {key} is not i64"))
    }

    fn attr_f(&self, name: &str, attr: &str) -> Option<f32> {
        self.nodes.get(name)?.attribute.iter().find(|a| a.name == attr).map(|a| a.f)
    }

    /// A Conv node's weight (and bias, zeros when absent) with its out channels.
    fn conv(&self, name: &str, w_len: usize, out_ch: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let w = self.f32_input(name, 1, w_len)?;
        let b = if self.node(name)?.input.len() > 2 && !self.node(name)?.input[2].is_empty() {
            self.f32_input(name, 2, out_ch)?
        } else {
            vec![0.0; out_ch]
        };
        Ok((w, b))
    }
}

// ---------------------------------------------------------------- recognizer

const C: usize = 128; // residual width
const E: usize = 256; // expanded width
const SE: usize = 64; // squeeze width
const DW: usize = 9; // temporal depthwise kernel
const CTX_C: usize = 32;
const CTX_H: usize = 64;
const CTX_G: usize = 4; // context pooling groups

struct Block {
    dw_w: Vec<f32>, // [9][C] (transposed from the graph's [C][9])
    dw_b: Vec<f32>,
    exp_w: Vec<f32>, // [E][C]
    exp_b: Vec<f32>,
    fc1_w: Vec<f32>, // [SE][E]
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>, // [E][SE]
    fc2_b: Vec<f32>,
    proj_w: Vec<f32>, // [C][E]
    proj_b: Vec<f32>,
}

/// The base recognizer's context branch: per-column soft pooling over the
/// three-line window.
struct Context {
    rows: usize,         // pooled rows (input height / 4)
    patch_w: Vec<f32>,   // [32][16]
    patch_b: Vec<f32>,
    dw_w: Vec<f32>,      // [9][32] (3×3, no bias; transposed)
    pw_w: Vec<f32>,      // [64][32]
    pw_b: Vec<f32>,
    score_w: Vec<f32>,   // [4][64]
    score_b: Vec<f32>,
    row_prior: Vec<f32>, // [4][rows]
    proj_w: Vec<f32>,    // [128][64]
    proj_b: Vec<f32>,
}

/// One Indra recognizer (`rec*.onnx`) written out by hand.
pub struct RecNet {
    height: usize,    // input height
    slice: (usize, usize), // stem rows
    stem_w: Vec<f32>, // [C][32*4]
    stem_b: Vec<f32>,
    bn_scale: Vec<f32>,
    bn_shift: Vec<f32>,
    context: Option<Context>,
    blocks: Vec<Block>,
    head_w: Vec<f32>, // [V][C]
    head_b: Vec<f32>,
    vocab: usize,
}

impl RecNet {
    /// Read a recognizer's weights; Err when the graph is not the one this
    /// code implements (the caller then keeps tract).
    pub fn load(path: &Path) -> Result<Self, String> {
        let g = Graph::load(path)?;
        let starts = g.i64_input("/rec/Slice", 1)?;
        let ends = g.i64_input("/rec/Slice", 2)?;
        let axes = g.i64_input("/rec/Slice", 3)?;
        if axes != [2] || starts.len() != 1 || ends.len() != 1 || ends[0] - starts[0] != 32 {
            return Err(format!("unexpected stem slice {starts:?}..{ends:?} on {axes:?}"));
        }
        let slice = (starts[0] as usize, ends[0] as usize);
        let (stem_w, stem_b) = g.conv("/rec/stem/Conv", C * 32 * 4, C)?;
        let bn = "/rec/stem_bn/BatchNormalization";
        let eps = g.attr_f(bn, "epsilon").unwrap_or(1e-5);
        let (gamma, beta, mean, var) = (g.f32_input(bn, 1, C)?, g.f32_input(bn, 2, C)?, g.f32_input(bn, 3, C)?, g.f32_input(bn, 4, C)?);
        let bn_scale: Vec<f32> = (0..C).map(|c| gamma[c] / (var[c] + eps).sqrt()).collect();
        let bn_shift: Vec<f32> = (0..C).map(|c| beta[c] - mean[c] * bn_scale[c]).collect();
        let context = if g.has("/rec/context/patch/Conv") {
            let key = g.node("/rec/context/Add")?.input.get(1).cloned().ok_or("context add")?;
            let rows_tensor = g.tensors.get(&key).and_then(proto_f32).ok_or("context row prior")?;
            if rows_tensor.is_empty() || rows_tensor.len() % CTX_G != 0 {
                return Err("context row prior shape".into());
            }
            let rows = rows_tensor.len() / CTX_G;
            let (patch_w, patch_b) = g.conv("/rec/context/patch/Conv", CTX_C * 16, CTX_C)?;
            let (dw_raw, dw_b) = g.conv("/rec/context/dw/Conv", CTX_C * 9, CTX_C)?;
            let dw_w: Vec<f32> = (0..9).flat_map(|k| (0..CTX_C).map(move |c| (k, c))).map(|(k, c)| dw_raw[c * 9 + k]).collect();
            if dw_b.iter().any(|&b| b != 0.0) {
                return Err("context depthwise has a bias".into());
            }
            let (pw_w, pw_b) = g.conv("/rec/context/pw/Conv", CTX_H * CTX_C, CTX_H)?;
            let (score_w, score_b) = g.conv("/rec/context/score/Conv", CTX_G * CTX_H, CTX_G)?;
            let (proj_w, proj_b) = g.conv("/rec/context/proj/Conv", C * CTX_H, C)?;
            Some(Context { rows, patch_w, patch_b, dw_w, pw_w, pw_b, score_w, score_b, row_prior: rows_tensor, proj_w, proj_b })
        } else {
            None
        };
        let height = match &context {
            Some(c) => c.rows * 4,
            None => slice.1,
        };
        let mut blocks = vec![];
        for i in 0.. {
            let p = format!("/rec/blocks.{i}");
            if !g.has(&format!("{p}/dw/Conv")) {
                break;
            }
            let (dw_raw, dw_b) = g.conv(&format!("{p}/dw/Conv"), C * DW, C)?;
            let dw_w: Vec<f32> = (0..DW).flat_map(|k| (0..C).map(move |c| (k, c))).map(|(k, c)| dw_raw[c * DW + k]).collect();
            let (exp_w, exp_b) = g.conv(&format!("{p}/expand/Conv"), E * C, E)?;
            let (fc1_w, fc1_b) = g.conv(&format!("{p}/se/fc1/Conv"), SE * E, SE)?;
            let (fc2_w, fc2_b) = g.conv(&format!("{p}/se/fc2/Conv"), E * SE, E)?;
            let (proj_w, proj_b) = g.conv(&format!("{p}/project/Conv"), C * E, C)?;
            blocks.push(Block { dw_w, dw_b, exp_w, exp_b, fc1_w, fc1_b, fc2_w, fc2_b, proj_w, proj_b });
        }
        if blocks.is_empty() {
            return Err("no blocks".into());
        }
        let head = g.node("/rec/head/Conv")?;
        let hw = g.tensors.get(&head.input[1]).ok_or("head weight")?;
        let vocab = *hw.dims.first().ok_or("head dims")? as usize;
        let (head_w, head_b) = g.conv("/rec/head/Conv", vocab * C, vocab)?;
        Ok(RecNet { height, slice, stem_w, stem_b, bn_scale, bn_shift, context, blocks, head_w, head_b, vocab })
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// The recognizer's outputs for `x [b, 1, h, w]` and `widths [b]`:
    /// `ids, conf, lengths` and, with `full`, `blank, topv, topi` (the same
    /// tensors as the ONNX graph's outputs).
    pub fn run(&self, b: usize, h: usize, w: usize, x: &[f32], widths: &[i64], full: bool) -> Result<Vec<Output>, ModelError> {
        let err = |m: String| ModelError::Run("rec (native)".into(), m);
        if h != self.height || x.len() != b * h * w || widths.len() != b || w < 4 {
            return Err(err(format!("bad input {b}x{h}x{w} for height {}", self.height)));
        }
        let t = (w - 4) / 4 + 1;
        let bt = b * t;
        let prof = std::env::var_os("KLEOS_FASTNN_PROF").is_some();
        let mut tk = std::time::Instant::now();
        let mut lap = |name: &str| {
            if prof {
                eprintln!("  fastnn {name}: {:.1} ms", tk.elapsed().as_secs_f64() * 1000.0);
                tk = std::time::Instant::now();
            }
        };
        // lengths = clip(ceil(width / 4), max T); the width mask
        let lengths: Vec<i64> = widths.iter().map(|&wd| ((wd + 3).div_euclid(4)).min(t as i64)).collect();
        let mask: Vec<f32> = (0..bt).map(|i| if ((i % t) as i64) < lengths[i / t] { 1.0 } else { 0.0 }).collect();

        // stem: 32×4 patches of the middle rows, stride (32, 4)
        let mut patches = vec![0f32; bt * 128];
        for bi in 0..b {
            for r in 0..32 {
                let row = &x[(bi * h + self.slice.0 + r) * w..][..w];
                for ti in 0..t {
                    let dst = &mut patches[(bi * t + ti) * 128 + r * 4..][..4];
                    dst.copy_from_slice(&row[ti * 4..ti * 4 + 4]);
                }
            }
        }
        let mut s = Vec::new();
        linear(bt, C, 128, &patches, &self.stem_w, &self.stem_b, &mut s);
        drop(patches);
        lap("stem");

        if let Some(cx) = &self.context {
            let proj = self.context(cx, b, h, w, t, x);
            for (a, p) in s.iter_mut().zip(proj.iter()) {
                *a += *p;
            }
        }
        lap("context");
        // batch norm, HardSwish, mask
        for (i, row) in s.chunks_exact_mut(C).enumerate() {
            let m = mask[i];
            for c in 0..C {
                row[c] = hardswish(row[c] * self.bn_scale[c] + self.bn_shift[c]) * m;
            }
        }
        let mut xs = s;
        let mut y = vec![0f32; bt * C];
        let mut e: Vec<f32> = Vec::new();
        let mut p: Vec<f32> = Vec::new();
        let counts: Vec<f32> = (0..b).map(|bi| (lengths[bi] as f32).max(1.0)).collect();
        for blk in &self.blocks {
            // depthwise k=9 along time, zero padded per sequence
            for bi in 0..b {
                let base = bi * t;
                for ti in 0..t {
                    y[(base + ti) * C..][..C].copy_from_slice(&blk.dw_b);
                }
                for k in 0..DW {
                    let wk = &blk.dw_w[k * C..][..C];
                    // out[ti] += w[k] * x[ti + k - 4] for the ti whose source is inside
                    let lo = 4usize.saturating_sub(k);
                    let hi = (t + 4).saturating_sub(k).min(t);
                    for ti in lo..hi {
                        let src = ti + k - 4;
                        let (out, inp) = (&mut y[(base + ti) * C..][..C], &xs[(base + src) * C..][..C]);
                        for c in 0..C {
                            out[c] += wk[c] * inp[c];
                        }
                    }
                }
            }
            // expand + HardSwish
            linear(bt, E, C, &y, &blk.exp_w, &blk.exp_b, &mut e);
            hardswish_inplace(&mut e);
            // masked squeeze-excite
            let mut pooled = vec![0f32; b * E];
            for bi in 0..b {
                let acc = &mut pooled[bi * E..][..E];
                for ti in 0..t {
                    if mask[bi * t + ti] == 0.0 {
                        continue;
                    }
                    let row = &e[(bi * t + ti) * E..][..E];
                    for c in 0..E {
                        acc[c] += row[c];
                    }
                }
                let inv = 1.0 / counts[bi];
                for v in acc.iter_mut() {
                    *v *= inv;
                }
            }
            let mut z = Vec::new();
            linear(b, SE, E, &pooled, &blk.fc1_w, &blk.fc1_b, &mut z);
            for v in z.iter_mut() {
                *v = v.max(0.0);
            }
            let mut gate = Vec::new();
            linear(b, E, SE, &z, &blk.fc2_w, &blk.fc2_b, &mut gate);
            for v in gate.iter_mut() {
                *v = sigmoid(*v);
            }
            for bi in 0..b {
                let gr = &gate[bi * E..][..E];
                for ti in 0..t {
                    let row = &mut e[(bi * t + ti) * E..][..E];
                    for c in 0..E {
                        row[c] *= gr[c];
                    }
                }
            }
            // project, residual, mask
            linear(bt, C, E, &e, &blk.proj_w, &blk.proj_b, &mut p);
            for (i, (xr, pr)) in xs.chunks_exact_mut(C).zip(p.chunks_exact(C)).enumerate() {
                let m = mask[i];
                for c in 0..C {
                    xr[c] = (xr[c] + pr[c]) * m;
                }
            }
        }
        drop((y, e, p));
        lap("blocks");

        // classifier: logits [b*t][V], already in the graph's [B, T, V] order
        let v = self.vocab;
        let mut logits = Vec::new();
        linear(bt, v, C, &xs, &self.head_w, &self.head_b, &mut logits);
        let mut ids = vec![0i64; bt];
        let mut conf = vec![0f32; bt];
        let (mut blank, mut topv, mut topi) = if full {
            (vec![0f32; bt], vec![0f32; bt * TOPK], vec![0i64; bt * TOPK])
        } else {
            (vec![], vec![], vec![])
        };
        let mut scratch = vec![0f32; v];
        for r in 0..bt {
            let row = &logits[r * v..][..v];
            let best = max_f32(row);
            let arg = row.iter().position(|&l| l == best).unwrap_or(0);
            for (d, &l) in scratch.iter_mut().zip(row.iter()) {
                *d = l - best;
            }
            blas::exp_inplace(&mut scratch);
            let sum = sum_f32(&scratch);
            ids[r] = arg as i64;
            conf[r] = 1.0 / sum;
            if full {
                let lse = best + sum.ln();
                blank[r] = row[0] - lse;
                // top-k by value, ties to the lower index (as ONNX Runtime)
                let mut top: [(f32, usize); TOPK] = [(f32::NEG_INFINITY, usize::MAX); TOPK];
                for (j, &l) in row.iter().enumerate() {
                    if l > top[TOPK - 1].0 {
                        let mut k = TOPK - 1;
                        while k > 0 && l > top[k - 1].0 {
                            top[k] = top[k - 1];
                            k -= 1;
                        }
                        top[k] = (l, j);
                    }
                }
                for (k, (l, j)) in top.iter().enumerate() {
                    topv[r * TOPK + k] = l - lse;
                    topi[r * TOPK + k] = *j as i64;
                }
            }
        }
        lap("head");
        let mut out = vec![Output::I64(vec![b, t], ids), Output::F32(vec![b, t], conf), Output::I64(vec![b], lengths)];
        if full {
            out.push(Output::F32(vec![b, t], blank));
            out.push(Output::F32(vec![b, t, TOPK], topv));
            out.push(Output::I64(vec![b, t, TOPK], topi));
        }
        Ok(out)
    }

    /// The context branch, projected to the residual width: `[b*t][C]`.
    fn context(&self, cx: &Context, b: usize, h: usize, w: usize, t: usize, x: &[f32]) -> Vec<f32> {
        let rows = cx.rows;
        let n = b * rows * t;
        // 4×4 patches, stride 4 → [b][row][t][32]
        let mut patches = vec![0f32; n * 16];
        for bi in 0..b {
            for hr in 0..rows {
                for r in 0..4 {
                    let src = &x[(bi * h + hr * 4 + r) * w..][..w];
                    for ti in 0..t {
                        let dst = &mut patches[((bi * rows + hr) * t + ti) * 16 + r * 4..][..4];
                        dst.copy_from_slice(&src[ti * 4..ti * 4 + 4]);
                    }
                }
            }
        }
        let mut a = Vec::new();
        linear(n, CTX_C, 16, &patches, &cx.patch_w, &cx.patch_b, &mut a);
        drop(patches);
        hardswish_inplace(&mut a);
        // depthwise 3×3, padding 1, no bias
        let mut d = vec![0f32; n * CTX_C];
        for bi in 0..b {
            for hr in 0..rows {
                for ti in 0..t {
                    let out = &mut d[((bi * rows + hr) * t + ti) * CTX_C..][..CTX_C];
                    for dh in 0..3 {
                        let hh = hr as isize + dh as isize - 1;
                        if hh < 0 || hh >= rows as isize {
                            continue;
                        }
                        for dt in 0..3 {
                            let tt = ti as isize + dt as isize - 1;
                            if tt < 0 || tt >= t as isize {
                                continue;
                            }
                            let src = &a[((bi * rows + hh as usize) * t + tt as usize) * CTX_C..][..CTX_C];
                            let wk = &cx.dw_w[(dh * 3 + dt) * CTX_C..][..CTX_C];
                            for c in 0..CTX_C {
                                out[c] += wk[c] * src[c];
                            }
                        }
                    }
                }
            }
        }
        drop(a);
        let mut h1 = Vec::new();
        linear(n, CTX_H, CTX_C, &d, &cx.pw_w, &cx.pw_b, &mut h1);
        drop(d);
        hardswish_inplace(&mut h1);
        let mut score = Vec::new();
        linear(n, CTX_G, CTX_H, &h1, &cx.score_w, &cx.score_b, &mut score);
        // softmax over rows per (b, group, t), then pool: pooled[b][t][g*16+k]
        let per = CTX_H / CTX_G;
        let mut pooled = vec![0f32; b * t * CTX_H];
        let mut wts = vec![0f32; rows];
        for bi in 0..b {
            for ti in 0..t {
                for g in 0..CTX_G {
                    let mut mx = f32::NEG_INFINITY;
                    for hr in 0..rows {
                        let v = score[((bi * rows + hr) * t + ti) * CTX_G + g] + cx.row_prior[g * rows + hr];
                        wts[hr] = v;
                        mx = mx.max(v);
                    }
                    let mut sum = 0f32;
                    for v in wts.iter_mut() {
                        *v = (*v - mx).exp();
                        sum += *v;
                    }
                    let inv = 1.0 / sum;
                    let acc = &mut pooled[(bi * t + ti) * CTX_H + g * per..][..per];
                    for hr in 0..rows {
                        let wv = wts[hr] * inv;
                        let src = &h1[((bi * rows + hr) * t + ti) * CTX_H + g * per..][..per];
                        for k in 0..per {
                            acc[k] += wv * src[k];
                        }
                    }
                }
            }
        }
        drop(h1);
        let mut proj = Vec::new();
        linear(b * t, C, CTX_H, &pooled, &cx.proj_w, &cx.proj_b, &mut proj);
        proj
    }
}
