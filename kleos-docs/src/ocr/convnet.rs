//! A small ONNX interpreter for Indra's 2-D convolution nets (the line
//! detector and the orientation model), with hand-written convolution
//! kernels.
//!
//! These graphs are MobileNet-style stacks (a 3×3 stride-2 stem, depthwise
//! 3×3 / 5×5, pointwise 1×1, HardSwish, residual adds) plus, in the detector,
//! a feature pyramid (nearest Resize, Add) and a pixel-shuffle head. tract runs
//! them correctly but slowly around the tiny convolutions; here
//! - a pointwise convolution is one GEMM per image (Accelerate on macOS),
//! - a depthwise or small dense convolution is direct multiply-adds over
//!   contiguous output rows,
//! - every other op is evaluated directly with ONNX semantics, in NCHW.
//!
//! Only the op set these graphs use is implemented; anything else is refused
//! at load and the caller keeps tract.

use std::collections::HashMap;
use std::path::Path;

use tract_onnx::pb::{AttributeProto, NodeProto, TensorProto};
use tract_onnx::prelude::Framework;

use super::fastnn::blas_gemm_nn;
use super::models::ModelError;

#[derive(Clone, Debug)]
enum T {
    F(Vec<usize>, Vec<f32>),
    I(Vec<usize>, Vec<i64>),
}

impl T {
    fn shape(&self) -> &[usize] {
        match self {
            T::F(s, _) | T::I(s, _) => s,
        }
    }
    fn ints(&self) -> Result<Vec<i64>, String> {
        match self {
            T::I(_, v) => Ok(v.clone()),
            T::F(_, v) => Ok(v.iter().map(|&x| x as i64).collect()),
        }
    }
}

fn tensor_of(t: &TensorProto) -> Result<T, String> {
    let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
    match t.data_type {
        1 => {
            let v: Vec<f32> = if !t.raw_data.is_empty() {
                t.raw_data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            } else {
                t.float_data.clone()
            };
            Ok(T::F(shape, v))
        }
        7 => {
            let v: Vec<i64> = if !t.raw_data.is_empty() {
                t.raw_data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
            } else {
                t.int64_data.clone()
            };
            Ok(T::I(shape, v))
        }
        d => Err(format!("tensor {} has data type {d}", t.name)),
    }
}

fn attr<'a>(n: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    n.attribute.iter().find(|a| a.name == name)
}

fn attr_ints(n: &NodeProto, name: &str) -> Option<Vec<i64>> {
    attr(n, name).map(|a| a.ints.clone())
}

fn attr_i(n: &NodeProto, name: &str, d: i64) -> i64 {
    attr(n, name).map(|a| a.i).unwrap_or(d)
}

fn attr_s(n: &NodeProto, name: &str) -> String {
    attr(n, name).map(|a| String::from_utf8_lossy(&a.s).to_string()).unwrap_or_default()
}

fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

#[inline(always)]
fn hardswish(x: f32) -> f32 {
    x * (x + 3.0).clamp(0.0, 6.0) * (1.0 / 6.0)
}

/// A convolution prepared at load.
struct ConvW {
    co: usize,
    ci: usize,
    group: usize,
    k: (usize, usize),
    s: (usize, usize),
    p: (usize, usize, usize, usize), // top, left, bottom, right
    w: Vec<f32>, // [co][ci/group][kh][kw]
    b: Vec<f32>,
    /// HardSwish fused into the output
    act: bool,
}

/// `out[oh×ow] += w * in[ih×iw]` shifted by kernel tap (ky, kx).
#[inline]
fn tap(out: &mut [f32], oh: usize, ow: usize, inp: &[f32], ih: usize, iw: usize, wv: f32, ky: usize, kx: usize, cw: &ConvW) {
    let (sy, sx) = cw.s;
    let (pt, pl) = (cw.p.0 as isize, cw.p.1 as isize);
    // valid output columns: 0 <= ox*sx + kx - pl < iw
    let off = kx as isize - pl;
    let ox_lo = if off >= 0 { 0 } else { ((-off) as usize).div_ceil(sx) };
    let ox_hi = if (iw as isize) - off <= 0 { 0 } else { (((iw as isize - off) as usize) + sx - 1) / sx }.min(ow);
    if ox_lo >= ox_hi {
        return;
    }
    for oy in 0..oh {
        let iy = (oy * sy) as isize + ky as isize - pt;
        if iy < 0 || iy >= ih as isize {
            continue;
        }
        let orow = &mut out[oy * ow..(oy + 1) * ow];
        let irow = &inp[iy as usize * iw..(iy as usize + 1) * iw];
        if sx == 1 {
            let ix0 = (ox_lo as isize + off) as usize;
            let n = ox_hi - ox_lo;
            let (o, i) = (&mut orow[ox_lo..ox_hi], &irow[ix0..ix0 + n]);
            for j in 0..n {
                o[j] += wv * i[j];
            }
        } else {
            for ox in ox_lo..ox_hi {
                let ix = (ox * sx) as isize + off;
                orow[ox] += wv * irow[ix as usize];
            }
        }
    }
}

fn conv(x: &T, cw: &ConvW) -> Result<T, String> {
    let T::F(sh, xv) = x else { return Err("conv on ints".into()) };
    if sh.len() != 4 || sh[1] != cw.ci {
        return Err(format!("conv input {sh:?}, expected {} channels", cw.ci));
    }
    let (n, ih, iw) = (sh[0], sh[2], sh[3]);
    let (kh, kw) = cw.k;
    let oh = (ih + cw.p.0 + cw.p.2 - kh) / cw.s.0 + 1;
    let ow = (iw + cw.p.1 + cw.p.3 - kw) / cw.s.1 + 1;
    let plane_o = oh * ow;
    let plane_i = ih * iw;
    let mut out = vec![0f32; n * cw.co * plane_o];
    let pointwise = kh == 1 && kw == 1 && cw.s == (1, 1) && cw.p == (0, 0, 0, 0) && cw.group == 1;
    for bi in 0..n {
        let xi = &xv[bi * cw.ci * plane_i..(bi + 1) * cw.ci * plane_i];
        let oo = &mut out[bi * cw.co * plane_o..(bi + 1) * cw.co * plane_o];
        for (c, row) in oo.chunks_exact_mut(plane_o).enumerate() {
            row.fill(cw.b[c]);
        }
        if pointwise {
            // out[co][hw] += w[co][ci] · x[ci][hw]
            blas_gemm_nn(cw.co, plane_o, cw.ci, &cw.w, xi, oo, true);
            continue;
        }
        if cw.group == 1 {
            // im2col: cols[ci*kh*kw][oh*ow], then out[co][hw] += w[co][·] · cols
            let kk = cw.ci * kh * kw;
            let mut cols = vec![0f32; kk * plane_o];
            for ci in 0..cw.ci {
                let plane = &xi[ci * plane_i..(ci + 1) * plane_i];
                for ky in 0..kh {
                    for kx in 0..kw {
                        let row = &mut cols[((ci * kh + ky) * kw + kx) * plane_o..][..plane_o];
                        tap(row, oh, ow, plane, ih, iw, 1.0, ky, kx, cw);
                    }
                }
            }
            blas_gemm_nn(cw.co, plane_o, kk, &cw.w, &cols, oo, true);
            continue;
        }
        let cig = cw.ci / cw.group;
        let cog = cw.co / cw.group;
        for co in 0..cw.co {
            let g = co / cog;
            let orow = &mut oo[co * plane_o..(co + 1) * plane_o];
            for cl in 0..cig {
                let ci = g * cig + cl;
                let plane = &xi[ci * plane_i..(ci + 1) * plane_i];
                for ky in 0..kh {
                    for kx in 0..kw {
                        let wv = cw.w[((co * cig + cl) * kh + ky) * kw + kx];
                        if wv != 0.0 {
                            tap(orow, oh, ow, plane, ih, iw, wv, ky, kx, cw);
                        }
                    }
                }
            }
        }
    }
    if cw.act {
        for v in out.iter_mut() {
            *v = hardswish(*v);
        }
    }
    Ok(T::F(vec![n, cw.co, oh, ow], out))
}

enum Op {
    Conv(ConvW),
    HardSwish,
    Relu,
    Sigmoid,
    Softplus,
    Add,
    Mul,
    Resize,
    DepthToSpace(usize),
    Slice,
    Shape,
    Gather(i64),
    Unsqueeze,
    Squeeze,
    Concat(i64),
    Cast(i32),
    Reshape,
    Transpose(Vec<usize>),
    ReduceMean { axes: Vec<i64>, keep: bool },
    Gemm { trans_b: bool },
    Softmax(i64),
}

struct Step {
    op: Op,
    inputs: Vec<String>,
    output: String,
}

/// An Indra 2-D convolution net run without tract.
pub struct ConvNet {
    name: String,
    consts: HashMap<String, T>,
    steps: Vec<Step>,
    input: String,
    outputs: Vec<String>,
    last_use: HashMap<String, usize>,
}

impl ConvNet {
    pub fn load(path: &Path) -> Result<Self, String> {
        let proto = tract_onnx::onnx().proto_model_for_path(path).map_err(|e| format!("{e:?}"))?;
        let g = proto.graph.ok_or("no graph")?;
        let mut consts: HashMap<String, T> = HashMap::new();
        for t in &g.initializer {
            consts.insert(t.name.clone(), tensor_of(t)?);
        }
        let inits: std::collections::HashSet<String> = g.initializer.iter().map(|t| t.name.clone()).collect();
        let input = g.input.iter().map(|i| i.name.clone()).find(|n| !inits.contains(n)).ok_or("no input")?;
        let outputs: Vec<String> = g.output.iter().map(|o| o.name.clone()).collect();
        let mut steps = vec![];
        for n in &g.node {
            let out = n.output.first().cloned().unwrap_or_default();
            let op = match n.op_type.as_str() {
                "Constant" => {
                    let t = attr(n, "value").and_then(|a| a.t.as_ref()).ok_or("constant without value")?;
                    consts.insert(out, tensor_of(t)?);
                    continue;
                }
                "Conv" => {
                    let Some(T::F(ws, w)) = consts.get(&n.input[1]).cloned() else { return Err(format!("{}: weight", n.name)) };
                    if ws.len() != 4 {
                        return Err(format!("{}: {}-d weight", n.name, ws.len()));
                    }
                    let group = attr_i(n, "group", 1) as usize;
                    let co = ws[0];
                    let b = match n.input.get(2).filter(|s| !s.is_empty()).and_then(|k| consts.get(k)) {
                        Some(T::F(_, b)) => b.clone(),
                        _ => vec![0.0; co],
                    };
                    if attr_ints(n, "dilations").map(|d| d.iter().any(|&x| x != 1)).unwrap_or(false) {
                        return Err(format!("{}: dilation", n.name));
                    }
                    let s = attr_ints(n, "strides").unwrap_or(vec![1, 1]);
                    let p = attr_ints(n, "pads").unwrap_or(vec![0, 0, 0, 0]);
                    if !attr_s(n, "auto_pad").is_empty() && attr_s(n, "auto_pad") != "NOTSET" {
                        return Err(format!("{}: auto_pad", n.name));
                    }
                    Op::Conv(ConvW {
                        co,
                        ci: ws[1] * group,
                        group,
                        k: (ws[2], ws[3]),
                        s: (s[0] as usize, s[1] as usize),
                        p: (p[0] as usize, p[1] as usize, p[2] as usize, p[3] as usize),
                        w,
                        b,
                        act: false,
                    })
                }
                "HardSwish" => Op::HardSwish,
                "Relu" => Op::Relu,
                "Sigmoid" => Op::Sigmoid,
                "Softplus" => Op::Softplus,
                "Add" => Op::Add,
                "Mul" => Op::Mul,
                "Resize" => {
                    if attr_s(n, "mode") != "nearest" {
                        return Err(format!("{}: resize mode {}", n.name, attr_s(n, "mode")));
                    }
                    let ctm = attr_s(n, "coordinate_transformation_mode");
                    let nm = attr_s(n, "nearest_mode");
                    if ctm != "asymmetric" || nm != "floor" {
                        return Err(format!("{}: resize {ctm}/{nm}", n.name));
                    }
                    Op::Resize
                }
                "DepthToSpace" => {
                    if attr_s(n, "mode") != "CRD" {
                        return Err(format!("{}: DepthToSpace {}", n.name, attr_s(n, "mode")));
                    }
                    Op::DepthToSpace(attr_i(n, "blocksize", 1) as usize)
                }
                "Slice" => Op::Slice,
                "Shape" => Op::Shape,
                "Gather" => Op::Gather(attr_i(n, "axis", 0)),
                "Unsqueeze" => Op::Unsqueeze,
                "Squeeze" => Op::Squeeze,
                "Concat" => Op::Concat(attr_i(n, "axis", 0)),
                "Cast" => Op::Cast(attr_i(n, "to", 1) as i32),
                "Reshape" => {
                    if attr_i(n, "allowzero", 0) != 0 {
                        return Err(format!("{}: allowzero", n.name));
                    }
                    Op::Reshape
                }
                "Transpose" => Op::Transpose(attr_ints(n, "perm").ok_or("transpose perm")?.iter().map(|&p| p as usize).collect()),
                "ReduceMean" => Op::ReduceMean { axes: attr_ints(n, "axes").ok_or("reducemean axes")?, keep: attr_i(n, "keepdims", 1) != 0 },
                "Gemm" => {
                    let (a, b) = (attr(n, "alpha").map(|a| a.f).unwrap_or(1.0), attr(n, "beta").map(|a| a.f).unwrap_or(1.0));
                    if a != 1.0 || b != 1.0 || attr_i(n, "transA", 0) != 0 {
                        return Err(format!("{}: gemm attributes", n.name));
                    }
                    Op::Gemm { trans_b: attr_i(n, "transB", 0) != 0 }
                }
                "Softmax" => Op::Softmax(attr_i(n, "axis", -1)),
                other => return Err(format!("{}: unsupported op {other}", n.name)),
            };
            steps.push(Step { op, inputs: n.input.clone(), output: out });
        }
        // fuse Conv → HardSwish when the conv output feeds only that op
        let mut uses: HashMap<String, usize> = HashMap::new();
        for st in &steps {
            for x in &st.inputs {
                *uses.entry(x.clone()).or_default() += 1;
            }
        }
        let mut fused: Vec<Step> = Vec::with_capacity(steps.len());
        let mut it = steps.into_iter().peekable();
        while let Some(mut st) = it.next() {
            if let Op::Conv(cw) = &mut st.op {
                if let Some(nx) = it.peek() {
                    if matches!(nx.op, Op::HardSwish) && nx.inputs.first() == Some(&st.output) && uses.get(&st.output) == Some(&1) && !outputs.contains(&st.output) {
                        cw.act = true;
                        st.output = nx.output.clone();
                        it.next();
                    }
                }
            }
            fused.push(st);
        }
        let steps = fused;
        let mut last_use = HashMap::new();
        for (i, s) in steps.iter().enumerate() {
            for x in &s.inputs {
                last_use.insert(x.clone(), i);
            }
        }
        Ok(ConvNet { name: path.display().to_string(), consts, steps, input, outputs, last_use })
    }

    /// One f32 input of `shape`; every output as (shape, data).
    pub fn run(&self, shape: &[usize], data: Vec<f32>) -> Result<Vec<(Vec<usize>, Vec<f32>)>, ModelError> {
        let err = |m: String| ModelError::Run(format!("{} (native)", self.name), m);
        let mut vals: HashMap<String, T> = HashMap::new();
        vals.insert(self.input.clone(), T::F(shape.to_vec(), data));
        let prof = std::env::var_os("KLEOS_FASTNN_PROF").is_some();
        let mut times: Vec<(String, f64)> = vec![];
        for (i, st) in self.steps.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let owned = matches!(st.op, Op::HardSwish | Op::Relu | Op::Sigmoid | Op::Softplus | Op::Add | Op::Mul)
                && self.last_use.get(&st.inputs[0]) == Some(&i)
                && !self.outputs.contains(&st.inputs[0])
                && vals.contains_key(&st.inputs[0]);
            let out = if owned {
                let x = vals.remove(&st.inputs[0]).unwrap();
                self.eval_inplace(st, x, &vals).map_err(|e| err(format!("{}: {e}", st.output)))?
            } else {
                self.eval(st, &vals).map_err(|e| err(format!("{}: {e}", st.output)))?
            };
            if prof {
                let name = match &st.op {
                    Op::Conv(c) if c.k == (1, 1) && c.group == 1 => "conv1x1".to_string(),
                    Op::Conv(c) if c.group > 1 => format!("dw{}x{}s{}", c.k.0, c.k.1, c.s.0),
                    Op::Conv(c) => format!("conv{}x{}s{}", c.k.0, c.k.1, c.s.0),
                    Op::HardSwish => "hardswish".into(),
                    Op::Add => "add".into(),
                    _ => "other".into(),
                };
                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                match times.iter_mut().find(|(n, _)| *n == name) {
                    Some(e) => e.1 += dt,
                    None => times.push((name, dt)),
                }
            }
            for x in &st.inputs {
                if self.last_use.get(x) == Some(&i) && !self.outputs.contains(x) {
                    vals.remove(x);
                }
            }
            vals.insert(st.output.clone(), out);
        }
        if prof {
            eprintln!("  convnet {}: {:?}", self.name.rsplit('/').next().unwrap_or(""), times.iter().map(|(n, t)| format!("{n} {t:.1}")).collect::<Vec<_>>());
        }
        self.outputs
            .iter()
            .map(|o| match vals.remove(o) {
                Some(T::F(s, v)) => Ok((s, v)),
                Some(T::I(s, v)) => Ok((s, v.into_iter().map(|x| x as f32).collect())),
                None => Err(err(format!("no output {o}"))),
            })
            .collect()
    }

    /// Unary ops and Add/Mul on a value that is not needed afterwards.
    fn eval_inplace(&self, st: &Step, x: T, vals: &HashMap<String, T>) -> Result<T, String> {
        let T::F(shape, mut v) = x else { return Err("in-place op on ints".into()) };
        match &st.op {
            Op::HardSwish => v.iter_mut().for_each(|x| *x = hardswish(*x)),
            Op::Relu => v.iter_mut().for_each(|x| *x = x.max(0.0)),
            Op::Sigmoid => v.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp())),
            Op::Softplus => v.iter_mut().for_each(|x| *x = if *x > 20.0 { *x } else { x.exp().ln_1p() }),
            Op::Add | Op::Mul => {
                let k = st.inputs.get(1).ok_or("missing input")?;
                let T::F(sb, b) = vals.get(k).or_else(|| self.consts.get(k)).ok_or_else(|| format!("missing value {k}"))? else { return Err("ints".into()) };
                let add = matches!(st.op, Op::Add);
                if *sb == shape {
                    if add {
                        v.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += *y);
                    } else {
                        v.iter_mut().zip(b.iter()).for_each(|(x, y)| *x *= *y);
                    }
                } else if b.len() == 1 {
                    let y = b[0];
                    v.iter_mut().for_each(|x| *x = if add { *x + y } else { *x * y });
                } else {
                    return Err(format!("broadcast {shape:?} {sb:?}"));
                }
            }
            _ => unreachable!(),
        }
        Ok(T::F(shape, v))
    }

    fn eval(&self, st: &Step, vals: &HashMap<String, T>) -> Result<T, String> {
        let inp = |i: usize| -> Result<&T, String> {
            let k = st.inputs.get(i).ok_or("missing input")?;
            vals.get(k).or_else(|| self.consts.get(k)).ok_or_else(|| format!("missing value {k}"))
        };
        let unary = |f: &dyn Fn(f32) -> f32| -> Result<T, String> {
            match inp(0)? {
                T::F(s, v) => Ok(T::F(s.clone(), v.iter().map(|&x| f(x)).collect())),
                _ => Err("unary on ints".into()),
            }
        };
        match &st.op {
            Op::Conv(cw) => conv(inp(0)?, cw),
            Op::HardSwish => unary(&hardswish),
            Op::Relu => unary(&|x| x.max(0.0)),
            Op::Sigmoid => unary(&|x| 1.0 / (1.0 + (-x).exp())),
            Op::Softplus => unary(&|x| if x > 20.0 { x } else { x.exp().ln_1p() }),
            Op::Add | Op::Mul => {
                let add = matches!(st.op, Op::Add);
                match (inp(0)?, inp(1)?) {
                    (T::F(sa, a), T::F(sb, b)) => {
                        let f = |x: f32, y: f32| if add { x + y } else { x * y };
                        if sa == sb {
                            Ok(T::F(sa.clone(), a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect()))
                        } else if b.len() == 1 {
                            Ok(T::F(sa.clone(), a.iter().map(|&x| f(x, b[0])).collect()))
                        } else if a.len() == 1 {
                            Ok(T::F(sb.clone(), b.iter().map(|&y| f(a[0], y)).collect()))
                        } else {
                            Err(format!("broadcast {sa:?} {sb:?}"))
                        }
                    }
                    (T::I(sa, a), T::I(_, b)) if a.len() == b.len() => Ok(T::I(sa.clone(), a.iter().zip(b.iter()).map(|(&x, &y)| if add { x + y } else { x * y }).collect())),
                    _ => Err("add/mul types".into()),
                }
            }
            Op::Resize => {
                let T::F(s, v) = inp(0)? else { return Err("resize ints".into()) };
                let sizes = inp(3)?.ints()?;
                if s.len() != 4 || sizes.len() != 4 || sizes[0] as usize != s[0] || sizes[1] as usize != s[1] {
                    return Err(format!("resize {s:?} to {sizes:?}"));
                }
                let (n, c, ih, iw) = (s[0], s[1], s[2], s[3]);
                let (oh, ow) = (sizes[2] as usize, sizes[3] as usize);
                // asymmetric + floor: x_in = floor(x_out * in / out)
                let ys: Vec<usize> = (0..oh).map(|y| ((y as f64 * ih as f64 / oh as f64).floor() as usize).min(ih - 1)).collect();
                let xs: Vec<usize> = (0..ow).map(|x| ((x as f64 * iw as f64 / ow as f64).floor() as usize).min(iw - 1)).collect();
                let mut out = vec![0f32; n * c * oh * ow];
                for p in 0..n * c {
                    let src = &v[p * ih * iw..(p + 1) * ih * iw];
                    let dst = &mut out[p * oh * ow..(p + 1) * oh * ow];
                    for (y, &sy) in ys.iter().enumerate() {
                        let srow = &src[sy * iw..(sy + 1) * iw];
                        let drow = &mut dst[y * ow..(y + 1) * ow];
                        for (x, &sx) in xs.iter().enumerate() {
                            drow[x] = srow[sx];
                        }
                    }
                }
                Ok(T::F(vec![n, c, oh, ow], out))
            }
            Op::DepthToSpace(bs) => {
                let T::F(s, v) = inp(0)? else { return Err("d2s ints".into()) };
                let (n, c, h, w) = (s[0], s[1], s[2], s[3]);
                let b = *bs;
                let co = c / (b * b);
                let mut out = vec![0f32; v.len()];
                for ni in 0..n {
                    for cc in 0..co {
                        for i in 0..b {
                            for j in 0..b {
                                let ci = cc * b * b + i * b + j; // CRD
                                let src = &v[((ni * c + ci) * h) * w..((ni * c + ci) * h + h) * w];
                                for y in 0..h {
                                    let drow = ((ni * co + cc) * h * b + y * b + i) * w * b;
                                    for x in 0..w {
                                        out[drow + x * b + j] = src[y * w + x];
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(T::F(vec![n, co, h * b, w * b], out))
            }
            Op::Slice => {
                let x = inp(0)?;
                let starts = inp(1)?.ints()?;
                let ends = inp(2)?.ints()?;
                let shape = x.shape().to_vec();
                let r = shape.len() as i64;
                let axes = if st.inputs.len() > 3 && !st.inputs[3].is_empty() { inp(3)?.ints()? } else { (0..starts.len() as i64).collect() };
                let steps = if st.inputs.len() > 4 && !st.inputs[4].is_empty() { inp(4)?.ints()? } else { vec![1; starts.len()] };
                if steps.iter().any(|&s| s != 1) {
                    return Err("slice step".into());
                }
                let mut lo = vec![0usize; shape.len()];
                let mut hi = shape.clone();
                for (k, &a) in axes.iter().enumerate() {
                    let a = if a < 0 { a + r } else { a } as usize;
                    let d = shape[a] as i64;
                    let cl = |v: i64| -> usize { (if v < 0 { v + d } else { v }).clamp(0, d) as usize };
                    lo[a] = cl(starts[k]);
                    hi[a] = cl(ends[k]).max(lo[a]);
                }
                let oshape: Vec<usize> = lo.iter().zip(hi.iter()).map(|(l, h)| h - l).collect();
                let ist = strides_of(&shape);
                let ost = strides_of(&oshape);
                let total: usize = oshape.iter().product();
                let r = shape.len();
                let run = if r == 0 { 1 } else { oshape[r - 1] };
                let starts_of = |o: usize| -> usize {
                    let mut rem = o;
                    let mut off = 0;
                    for d in 0..r {
                        let q = rem / ost[d];
                        rem %= ost[d];
                        off += (q + lo[d]) * ist[d];
                    }
                    off
                };
                match x {
                    T::F(_, v) => {
                        let mut out = Vec::with_capacity(total);
                        let mut o = 0;
                        while o < total {
                            let s0 = starts_of(o);
                            out.extend_from_slice(&v[s0..s0 + run]);
                            o += run.max(1);
                        }
                        Ok(T::F(oshape, out))
                    }
                    T::I(_, v) => Ok(T::I(oshape.clone(), (0..total).map(|o| v[starts_of(o)]).collect())),
                }
            }
            Op::Shape => {
                let s = inp(0)?.shape();
                Ok(T::I(vec![s.len()], s.iter().map(|&d| d as i64).collect()))
            }
            Op::Gather(axis) => {
                let data = inp(0)?.ints()?;
                if *axis != 0 || inp(0)?.shape().len() != 1 {
                    return Err("gather only on 1-d ints".into());
                }
                let idx = inp(1)?;
                let n = data.len() as i64;
                let pick = |i: i64| data[(if i < 0 { i + n } else { i }) as usize];
                Ok(T::I(idx.shape().to_vec(), idx.ints()?.into_iter().map(pick).collect()))
            }
            Op::Unsqueeze | Op::Squeeze => {
                let x = inp(0)?;
                let axes = inp(1)?.ints()?;
                let mut s = x.shape().to_vec();
                if matches!(st.op, Op::Unsqueeze) {
                    let r = (s.len() + axes.len()) as i64;
                    let mut ax: Vec<usize> = axes.iter().map(|&a| (if a < 0 { a + r } else { a }) as usize).collect();
                    ax.sort();
                    for a in ax {
                        s.insert(a, 1);
                    }
                } else {
                    let r = s.len() as i64;
                    let mut ax: Vec<usize> = axes.iter().map(|&a| (if a < 0 { a + r } else { a }) as usize).collect();
                    ax.sort();
                    for a in ax.into_iter().rev() {
                        if s[a] != 1 {
                            return Err("squeeze a non-1 axis".into());
                        }
                        s.remove(a);
                    }
                }
                Ok(match x {
                    T::F(_, v) => T::F(s, v.clone()),
                    T::I(_, v) => T::I(s, v.clone()),
                })
            }
            Op::Concat(axis) => {
                let parts: Vec<&T> = (0..st.inputs.len()).map(inp).collect::<Result<_, _>>()?;
                let s0 = parts[0].shape().to_vec();
                let r = s0.len() as i64;
                let a = (if *axis < 0 { axis + r } else { *axis }) as usize;
                let outer: usize = s0[..a].iter().product();
                let mut oshape = s0.clone();
                oshape[a] = parts.iter().map(|p| p.shape()[a]).sum();
                let inner = |p: &T| -> usize { p.shape()[a..].iter().product() };
                if parts.iter().all(|p| matches!(p, T::F(..))) {
                    let mut out = Vec::with_capacity(oshape.iter().product());
                    for o in 0..outer {
                        for p in &parts {
                            let T::F(_, v) = p else { unreachable!() };
                            let k = inner(p);
                            out.extend_from_slice(&v[o * k..(o + 1) * k]);
                        }
                    }
                    Ok(T::F(oshape, out))
                } else {
                    let mut out = Vec::new();
                    for o in 0..outer {
                        for p in &parts {
                            let v = p.ints()?;
                            let k = inner(p);
                            out.extend_from_slice(&v[o * k..(o + 1) * k]);
                        }
                    }
                    Ok(T::I(oshape, out))
                }
            }
            Op::Cast(to) => {
                let x = inp(0)?;
                match to {
                    7 | 6 => Ok(T::I(x.shape().to_vec(), x.ints()?)),
                    1 => Ok(match x {
                        T::F(s, v) => T::F(s.clone(), v.clone()),
                        T::I(s, v) => T::F(s.clone(), v.iter().map(|&x| x as f32).collect()),
                    }),
                    t => Err(format!("cast to {t}")),
                }
            }
            Op::Reshape => {
                let x = inp(0)?;
                let spec = inp(1)?.ints()?;
                let s = x.shape();
                let total: usize = s.iter().product();
                let mut out: Vec<i64> = spec.iter().enumerate().map(|(i, &d)| if d == 0 { s[i] as i64 } else { d }).collect();
                if let Some(pos) = out.iter().position(|&d| d == -1) {
                    let known: i64 = out.iter().filter(|&&d| d != -1).product();
                    out[pos] = total as i64 / known.max(1);
                }
                let os: Vec<usize> = out.iter().map(|&d| d as usize).collect();
                if os.iter().product::<usize>() != total {
                    return Err(format!("reshape {s:?} to {spec:?}"));
                }
                Ok(match x {
                    T::F(_, v) => T::F(os, v.clone()),
                    T::I(_, v) => T::I(os, v.clone()),
                })
            }
            Op::Transpose(perm) => {
                let T::F(s, v) = inp(0)? else { return Err("transpose ints".into()) };
                let os: Vec<usize> = perm.iter().map(|&p| s[p]).collect();
                let ist = strides_of(s);
                let ost = strides_of(&os);
                let r = os.len();
                let src_st: Vec<usize> = (0..r).map(|d| ist[perm[d]]).collect();
                let mut out = Vec::with_capacity(v.len());
                let mut idx = vec![0usize; r];
                let mut off = 0usize;
                let _ = &ost;
                for _ in 0..v.len() {
                    out.push(v[off]);
                    // advance the output multi-index (last axis fastest)
                    let mut d = r;
                    while d > 0 {
                        d -= 1;
                        idx[d] += 1;
                        off += src_st[d];
                        if idx[d] < os[d] {
                            break;
                        }
                        off -= src_st[d] * os[d];
                        idx[d] = 0;
                    }
                }
                Ok(T::F(os, out))
            }
            Op::ReduceMean { axes, keep } => {
                let T::F(s, v) = inp(0)? else { return Err("reducemean ints".into()) };
                let r = s.len() as i64;
                let ax: Vec<usize> = axes.iter().map(|&a| (if a < 0 { a + r } else { a }) as usize).collect();
                // only trailing contiguous axes (H, W of NCHW)
                let first = *ax.iter().min().unwrap();
                if ax.len() != s.len() - first || (first..s.len()).any(|d| !ax.contains(&d)) {
                    return Err("reducemean over non-trailing axes".into());
                }
                let inner: usize = s[first..].iter().product();
                let outer = v.len() / inner.max(1);
                let out: Vec<f32> = (0..outer).map(|o| v[o * inner..(o + 1) * inner].iter().sum::<f32>() / inner as f32).collect();
                let mut os = s[..first].to_vec();
                if *keep {
                    os.extend(std::iter::repeat(1).take(s.len() - first));
                }
                Ok(T::F(os, out))
            }
            Op::Gemm { trans_b } => {
                let (T::F(sa, a), T::F(sb, b)) = (inp(0)?, inp(1)?) else { return Err("gemm ints".into()) };
                let (m, k) = (sa[0], sa[1]);
                let n = if *trans_b { sb[0] } else { sb[1] };
                let mut out = vec![0f32; m * n];
                if let Ok(T::F(_, c)) = inp(2) {
                    for r in 0..m {
                        for j in 0..n {
                            out[r * n + j] = if c.len() == n { c[j] } else { c[r * n + j] };
                        }
                    }
                }
                for r in 0..m {
                    for j in 0..n {
                        let mut acc = 0f32;
                        for q in 0..k {
                            acc += a[r * k + q] * if *trans_b { b[j * k + q] } else { b[q * n + j] };
                        }
                        out[r * n + j] += acc;
                    }
                }
                Ok(T::F(vec![m, n], out))
            }
            Op::Softmax(axis) => {
                let T::F(s, v) = inp(0)? else { return Err("softmax ints".into()) };
                let r = s.len() as i64;
                if (if *axis < 0 { axis + r } else { *axis }) != r - 1 {
                    return Err("softmax not on the last axis".into());
                }
                let k = *s.last().unwrap();
                let mut out = v.clone();
                for row in out.chunks_exact_mut(k) {
                    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for x in row.iter_mut() {
                        *x = (*x - m).exp();
                        sum += *x;
                    }
                    for x in row.iter_mut() {
                        *x /= sum;
                    }
                }
                Ok(T::F(s.clone(), out))
            }
        }
    }
}
