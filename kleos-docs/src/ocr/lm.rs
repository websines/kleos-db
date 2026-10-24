//! The token n-gram language model (`lm.npz`) and the CTC prefix beam search
//! that uses it on lines the recognizer is unsure of (`lm.py`).

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

const TAG_SHIFT: u32 = 61;

/// A NumPy array read from a `.npy` member: little-endian u64, i64, f32 or f64.
#[derive(Debug)]
pub enum Npy {
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Npy {
    pub fn as_f64(&self) -> Vec<f64> {
        match self {
            Npy::U64(v) => v.iter().map(|&x| x as f64).collect(),
            Npy::I64(v) => v.iter().map(|&x| x as f64).collect(),
            Npy::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Npy::F64(v) => v.clone(),
        }
    }
    pub fn as_u64(&self) -> Vec<u64> {
        match self {
            Npy::U64(v) => v.clone(),
            Npy::I64(v) => v.iter().map(|&x| x as u64).collect(),
            Npy::F32(v) => v.iter().map(|&x| x as u64).collect(),
            Npy::F64(v) => v.iter().map(|&x| x as u64).collect(),
        }
    }
    pub fn scalar(&self) -> f64 {
        self.as_f64().first().copied().unwrap_or(0.0)
    }
}

pub fn parse_npy(b: &[u8]) -> Result<Npy, String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("not an npy array".into());
    }
    let (hlen, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let header = std::str::from_utf8(&b[off..off + hlen]).map_err(|e| e.to_string())?;
    let descr = header.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).ok_or("npy without descr")?;
    if header.contains("'fortran_order': True") {
        return Err("fortran-order npy".into());
    }
    let data = &b[off + hlen..];
    Ok(match descr {
        "<u8" => Npy::U64(data.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()),
        "<i8" => Npy::I64(data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()),
        "<i4" => Npy::I64(data.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect()),
        "<f4" => Npy::F32(data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
        "<f8" => Npy::F64(data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()),
        d => return Err(format!("npy dtype {d} not supported")),
    })
}

/// Every member of an `.npz` (a zip of `.npy` files), by name without `.npy`.
pub fn read_npz(path: &Path) -> Result<HashMap<String, Npy>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| e.to_string())?;
    let mut out = HashMap::new();
    for i in 0..z.len() {
        let mut m = z.by_index(i).map_err(|e| e.to_string())?;
        let name = m.name().trim_end_matches(".npy").to_string();
        let mut buf = Vec::with_capacity(m.size() as usize);
        m.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        out.insert(name, parse_npy(&buf)?);
    }
    Ok(out)
}

/// A backoff n-gram table over recognizer tokens.
pub struct NgramLm {
    pub order: usize,
    pub vocab_size: usize,
    pub bits: u32,
    /// per history length n: sorted (history, token) keys and log-probabilities
    k: Vec<Vec<u64>>,
    v: Vec<Vec<f64>>,
    /// per history length n: sorted history keys and backoff weights
    hk: Vec<Vec<u64>>,
    hb: Vec<Vec<f64>>,
    max_logp: f64,
    cache: std::sync::Mutex<HashMap<u64, f64>>,
}

fn sorted_pairs(k: Vec<u64>, v: Vec<f64>) -> (Vec<u64>, Vec<f64>) {
    if k.windows(2).all(|w| w[0] <= w[1]) {
        return (k, v);
    }
    let mut idx: Vec<usize> = (0..k.len()).collect();
    idx.sort_by_key(|&i| k[i]);
    (idx.iter().map(|&i| k[i]).collect(), idx.iter().map(|&i| v[i]).collect())
}

impl NgramLm {
    pub fn load(path: &Path) -> Result<Self, String> {
        let z = read_npz(path)?;
        let order = z.get("order").ok_or("lm without order")?.scalar() as usize;
        let vocab_size = z.get("vocab_size").ok_or("lm without vocab_size")?.scalar() as usize;
        let bits = z.get("bits").map(|b| b.scalar() as u32).unwrap_or(11);
        if order as u32 * bits > TAG_SHIFT {
            return Err(format!("order {order} x {bits} bits do not fit a key"));
        }
        let (mut k, mut v, mut hk, mut hb) = (vec![], vec![], vec![], vec![]);
        for n in 0..order {
            let (a, b) = sorted_pairs(z[&format!("k{n}")].as_u64(), z[&format!("v{n}")].as_f64());
            k.push(a);
            v.push(b);
            let (a, b) = sorted_pairs(z[&format!("hk{n}")].as_u64(), z[&format!("hb{n}")].as_f64());
            hk.push(a);
            hb.push(b);
        }
        let top = v.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max).max(-(vocab_size as f64).ln());
        let bo: f64 = hb.iter().filter(|h| !h.is_empty()).map(|h| h.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)).sum();
        Ok(NgramLm { order, vocab_size, bits, k, v, hk, hb, max_logp: top + bo, cache: Default::default() })
    }

    pub fn max_logp(&self) -> f64 {
        self.max_logp
    }

    fn find(keys: &[u64], key: u64) -> Option<usize> {
        keys.binary_search(&key).ok()
    }

    /// `logp_keys` for one key: the last order - 1 history tokens and the
    /// token, packed `bits` bits each.
    pub fn logp_key(&self, full: u64) -> f64 {
        if let Some(&v) = self.cache.lock().unwrap().get(&full) {
            return v;
        }
        let n_ord = self.order;
        let b = self.bits as u64;
        let ct = full & ((1u64 << b) - 1);
        let mut h_ok = vec![false; n_ord];
        let mut hbv = vec![0f64; n_ord];
        let mut p_ok = vec![false; n_ord];
        let mut pv = vec![0f64; n_ord];
        for n in 0..n_ord {
            let mask = if n == 0 { 0 } else { (1u64 << (b * n as u64)) - 1 };
            let hq = (full >> b) & mask;
            let pq = (hq << b) | ct;
            if let Some(j) = Self::find(&self.hk[n], hq) {
                h_ok[n] = true;
                hbv[n] = self.hb[n][j];
            }
            if let Some(j) = Self::find(&self.k[n], pq) {
                p_ok[n] = true;
                pv[n] = self.v[n][j];
            }
        }
        h_ok[0] = true;
        let mut m = 0;
        for n in 1..n_ord {
            if h_ok[n] {
                m += 1;
            } else {
                break;
            }
        }
        let mut suffix = vec![0f64; n_ord + 1];
        for n in (0..n_ord).rev() {
            suffix[n] = suffix[n + 1] + if n <= m { hbv[n] } else { 0.0 };
        }
        let chosen = (0..n_ord).rev().find(|&n| n <= m && p_ok[n]);
        let res = match chosen {
            Some(c) => suffix[c + 1] + pv[c],
            None => suffix[0] - (self.vocab_size as f64).ln(),
        };
        let mut c = self.cache.lock().unwrap();
        if c.len() < 1_000_000 {
            c.insert(full, res);
        }
        res
    }
}

#[inline]
fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[derive(Clone)]
struct Beam {
    b: f64,
    nb: f64,
    s: f64,
    steps: Vec<usize>,
    hk: u64,
}

/// Insertion-ordered map of prefixes (Python dict semantics).
#[derive(Default)]
struct Ordered {
    keys: Vec<Vec<u32>>,
    vals: Vec<Beam>,
    pos: HashMap<Vec<u32>, usize>,
}

impl Ordered {
    fn add(&mut self, prefix: Vec<u32>, b: f64, nb: f64, s: f64, st: &[usize], hk: u64) {
        if let Some(&i) = self.pos.get(&prefix) {
            let o = &mut self.vals[i];
            o.b = lse(o.b, b);
            o.nb = lse(o.nb, nb);
        } else {
            self.pos.insert(prefix.clone(), self.keys.len());
            self.keys.push(prefix);
            self.vals.push(Beam { b, nb, s, steps: st.to_vec(), hk });
        }
    }
    fn contains(&self, p: &[u32]) -> bool {
        self.pos.contains_key(p)
    }
}

/// CTC prefix beam search with the LM.  blank: [T] blank log-probabilities;
/// topv / topi: [T][k] the k best classes per step and their log-probabilities.
/// Returns (token ids, the step where each token was first emitted).
#[allow(clippy::too_many_arguments)]
pub fn beam_search(
    blank: &[f32],
    topv: &[Vec<f32>],
    topi: &[Vec<i64>],
    lm: &NgramLm,
    alpha: f64,
    beta: f64,
    beam: usize,
    blank_skip: f64,
    cand_floor: f64,
) -> (Vec<usize>, Vec<usize>) {
    let bits = lm.bits;
    let hmask: u64 = (1u64 << (bits as u64 * (lm.order as u64 - 1))) - 1;
    let mut beams = Ordered::default();
    beams.add(vec![], 0.0, f64::NEG_INFINITY, 0.0, &[], 0);
    let log_skip = blank_skip.ln();
    let log_floor = cand_floor.ln();
    let reach = beta + alpha * lm.max_logp();
    for t in 0..blank.len() {
        let b0 = blank[t] as f64;
        if b0 > log_skip {
            let mut nb = Ordered::default();
            for (p, e) in beams.keys.iter().zip(beams.vals.iter()) {
                nb.add(p.clone(), lse(e.b, e.nb) + b0, f64::NEG_INFINITY, e.s, &e.steps, e.hk);
            }
            beams = nb;
            continue;
        }
        let mut nxt = Ordered::default();
        let cands: Vec<(u32, f64)> = topi[t]
            .iter()
            .zip(topv[t].iter())
            .filter(|(&c, &v)| c != 0 && (v as f64) >= log_floor)
            .map(|(&c, &v)| (c as u32, v as f64))
            .collect();
        for (prefix, e) in beams.keys.iter().zip(beams.vals.iter()) {
            nxt.add(prefix.clone(), lse(e.b, e.nb) + b0, f64::NEG_INFINITY, e.s, &e.steps, e.hk);
            let last = prefix.last().copied();
            for &(c, lp) in &cands {
                if Some(c) == last {
                    nxt.add(prefix.clone(), f64::NEG_INFINITY, e.nb + lp, e.s, &e.steps, e.hk);
                }
            }
        }
        let floor = if beams.keys.len() >= beam {
            nxt.vals.iter().map(|e| lse(e.b, e.nb) + e.s).fold(f64::INFINITY, f64::min)
        } else {
            f64::NEG_INFINITY
        };
        let mut ext: Vec<(usize, u32, f64)> = vec![]; // (beam index, class, new_from)
        for (bi, (prefix, e)) in beams.keys.iter().zip(beams.vals.iter()).enumerate() {
            let tot = lse(e.b, e.nb);
            let last = prefix.last().copied();
            for &(c, lp) in &cands {
                let new_from = if Some(c) == last { e.b + lp } else { tot + lp };
                if new_from == f64::NEG_INFINITY {
                    continue;
                }
                let mut q = prefix.clone();
                q.push(c);
                if nxt.contains(&q) {
                    nxt.add(q, f64::NEG_INFINITY, new_from, e.s, &e.steps, e.hk);
                } else if new_from + e.s + reach >= floor - 1e-9 {
                    ext.push((bi, c, new_from));
                }
            }
        }
        for (bi, c, new_from) in ext {
            let e = &beams.vals[bi];
            let lmp = lm.logp_key((e.hk << bits) | (c as u64 + 1));
            let mut q = beams.keys[bi].clone();
            q.push(c);
            let mut st = e.steps.clone();
            st.push(t);
            nxt.add(q, f64::NEG_INFINITY, new_from, e.s + alpha * lmp + beta, &st, ((e.hk << bits) | (c as u64 + 1)) & hmask);
        }
        // stable sort by score, keep the best `beam`
        let mut idx: Vec<usize> = (0..nxt.keys.len()).collect();
        let score = |i: usize| -(lse(nxt.vals[i].b, nxt.vals[i].nb) + nxt.vals[i].s);
        idx.sort_by(|&a, &b| score(a).partial_cmp(&score(b)).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept = Ordered::default();
        for &i in idx.iter().take(beam) {
            let e = &nxt.vals[i];
            kept.pos.insert(nxt.keys[i].clone(), kept.keys.len());
            kept.keys.push(nxt.keys[i].clone());
            kept.vals.push(e.clone());
        }
        beams = kept;
    }
    let mut best = 0;
    let mut best_v = f64::NEG_INFINITY;
    for (i, e) in beams.vals.iter().enumerate() {
        let v = lse(e.b, e.nb) + e.s;
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    (beams.keys[best].iter().map(|&c| c as usize).collect(), beams.vals[best].steps.clone())
}
