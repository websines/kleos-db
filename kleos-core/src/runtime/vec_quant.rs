/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * The rotation (globally permuted block Walsh-Hadamard, two rounds), the
 * Lloyd-Max codebook fitted to the rotated-coordinate distribution, and the
 * length-renormalised inner-product estimator follow the design of turbovec
 * (https://github.com/RyanCodrai/turbovec, MIT, Copyright (c) 2026 Ryan
 * Codrai), itself an implementation of TurboQuant (arXiv:2504.19874). See
 * NOTICE.md.
 */

//! Compressed vector codes (TurboQuant-style) for HNSW traversal.
//!
//! A vector `v` is encoded as:
//!
//! 1. `u = v / ||v||`, zero-padded to `padded_dim` (a multiple of 64);
//! 2. `y = R u` for a fixed, seed-derived orthogonal rotation `R` (two rounds
//!    of global permutation + random sign flip + normalised block
//!    Walsh-Hadamard). After rotation every coordinate is distributed close to
//!    `N(0, 1/padded_dim)` regardless of the input data, so one
//!    data-independent scalar quantiser fits every coordinate;
//! 3. each `y_i` is replaced by the index of its Lloyd-Max cell (`bits` bits),
//!    the cells being optimal for that known distribution;
//! 4. two `f32`s are kept: `||v||`, and `||v|| / <y, ŷ>` which makes the
//!    inner-product estimate `<q, v> ≈ ||v|| <Rq, ŷ> / <y, ŷ>` exact for
//!    `q = v` (turbovec's length renormalisation, removing the downward bias
//!    of scalar quantisation).
//!
//! Queries are not quantised: `Rq` is computed once per search and scored
//! against codes with a per-byte decode table (asymmetric distance).
//!
//! The rotation and the bit layout are part of the on-disk format: **do not
//! change** [`ROTATION_SEED`], the PRNG, the round count, the padding rule, or
//! the packing order without introducing a new [`VecQuant`] variant.
//!
//! This module is self-contained (no crate-internal imports) so the codec can
//! be evaluated outside the database.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Quantisation applied to the vectors of an HNSW index for traversal.
#[derive(
    Debug, Copy, Clone, Default, PartialEq, Eq, Hash, serde_derive::Serialize, serde_derive::Deserialize,
)]
pub enum VecQuant {
    /// Full-precision traversal (the historical behaviour).
    #[default]
    None,
    /// 1 bit per rotated coordinate (sign).
    Binary,
    /// 2 bits per rotated coordinate.
    Turbo2,
    /// 4 bits per rotated coordinate.
    Turbo4,
    /// 8 bits per rotated coordinate.
    Turbo8,
}

impl VecQuant {
    /// Parse the user-facing option value.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "none" => VecQuant::None,
            "binary" | "turbo1" => VecQuant::Binary,
            "turbo2" => VecQuant::Turbo2,
            "turbo4" => VecQuant::Turbo4,
            "turbo8" => VecQuant::Turbo8,
            _ => return None,
        })
    }
    /// The user-facing option value.
    pub fn name(&self) -> &'static str {
        match self {
            VecQuant::None => "none",
            VecQuant::Binary => "binary",
            VecQuant::Turbo2 => "turbo2",
            VecQuant::Turbo4 => "turbo4",
            VecQuant::Turbo8 => "turbo8",
        }
    }
    /// Bits per rotated coordinate, `None` for full precision.
    pub fn bits(&self) -> Option<u32> {
        match self {
            VecQuant::None => None,
            VecQuant::Binary => Some(1),
            VecQuant::Turbo2 => Some(2),
            VecQuant::Turbo4 => Some(4),
            VecQuant::Turbo8 => Some(8),
        }
    }
}

/// Size of the per-code header: `||v||/<y,ŷ>` and `||v||`, both `f32` LE.
pub const CODE_HEADER: usize = 8;

/// Frozen rotation seed (part of the on-disk format).
const ROTATION_SEED: u64 = 0x6b6c_656f_735f_7471; // "kleos_tq"
/// Frozen number of permute/sign/Hadamard rounds.
const ROUNDS: usize = 2;

/// SplitMix64: tiny, portable, and frozen here so the rotation never depends
/// on an external crate's stream.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Dimension after zero padding: the next multiple of 64, so the Hadamard
/// blocks have at least 64 coordinates and packed codes need no tail.
pub fn padded_dim(dim: usize) -> usize {
    dim.max(1).div_ceil(64) * 64
}

/// Deterministic orthogonal rotation of `padded_dim` coordinates.
#[derive(Debug, Clone)]
pub struct Rotation {
    dim: usize,
    block: usize,
    inv_sqrt_block: f32,
    signs: Vec<Vec<f32>>,
    perms: Vec<Vec<u32>>,
}

impl Rotation {
    /// Rotation for vectors of (padded) dimension `dim`, a multiple of 64.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0 && dim % 64 == 0, "rotation dim must be a multiple of 64");
        let block = dim & dim.wrapping_neg();
        let mut rng = SplitMix64(ROTATION_SEED ^ (dim as u64).wrapping_mul(0x2545_f491_4f6c_dd1d));
        let mut signs = Vec::with_capacity(ROUNDS);
        let mut perms = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            signs.push(
                (0..dim)
                    .map(|_| if rng.next_u64() >> 63 == 1 { -1.0 } else { 1.0 })
                    .collect(),
            );
            let mut perm: Vec<u32> = (0..dim as u32).collect();
            for i in (1..dim).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                perm.swap(i, j);
            }
            perms.push(perm);
        }
        Self {
            dim,
            block,
            inv_sqrt_block: 1.0 / (block as f32).sqrt(),
            signs,
            perms,
        }
    }

    /// Rotate `row` (length `dim`) in place; `scratch` must have length `dim`.
    pub fn apply(&self, row: &mut [f32], scratch: &mut [f32]) {
        debug_assert_eq!(row.len(), self.dim);
        debug_assert_eq!(scratch.len(), self.dim);
        for round in 0..ROUNDS {
            let perm = &self.perms[round];
            let sign = &self.signs[round];
            for i in 0..self.dim {
                scratch[i] = row[perm[i] as usize] * sign[i];
            }
            for blk in scratch.chunks_exact_mut(self.block) {
                wht(blk);
                for x in blk.iter_mut() {
                    *x *= self.inv_sqrt_block;
                }
            }
            row.copy_from_slice(scratch);
        }
    }
}

/// Unnormalised in-place Walsh-Hadamard butterfly (length a power of two).
fn wht(blk: &mut [f32]) {
    let n = blk.len();
    let mut len = 1;
    while len < n {
        for chunk in blk.chunks_exact_mut(2 * len) {
            let (a, b) = chunk.split_at_mut(len);
            for (x, y) in a.iter_mut().zip(b.iter_mut()) {
                let (s, d) = (*x + *y, *x - *y);
                *x = s;
                *y = d;
            }
        }
        len <<= 1;
    }
}

/// Lloyd-Max scalar quantiser for one coordinate of a uniformly random unit
/// vector in `dim` dimensions (density ∝ (1 - x²)^((dim-3)/2), i.e. close to
/// `N(0, 1/dim)`). Returns `(boundaries, centroids)`, both ascending.
pub fn lloyd_max(bits: u32, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let levels = 1usize << bits;
    let sigma = 1.0 / (dim as f64).sqrt();
    let half_width = (12.0 * sigma).min(1.0);
    const GRID: usize = 1 << 16;
    let expo = (dim as f64 - 3.0).max(0.0) / 2.0;
    let xs: Vec<f64> = (0..GRID)
        .map(|k| -half_width + 2.0 * half_width * (k as f64 + 0.5) / GRID as f64)
        .collect();
    // Prefix sums of mass and first moment over the grid.
    let mut cum_w = Vec::with_capacity(GRID + 1);
    let mut cum_wx = Vec::with_capacity(GRID + 1);
    cum_w.push(0.0f64);
    cum_wx.push(0.0f64);
    for &x in &xs {
        let w = (expo * (1.0 - x * x).max(1e-300).ln()).exp();
        cum_w.push(cum_w.last().unwrap() + w);
        cum_wx.push(cum_wx.last().unwrap() + w * x);
    }
    let total = *cum_w.last().unwrap();
    // Initialise centroids at the distribution's quantile midpoints.
    let mut centroids: Vec<f64> = (0..levels)
        .map(|i| {
            let target = total * (i as f64 + 0.5) / levels as f64;
            let k = cum_w.partition_point(|&c| c < target).clamp(1, GRID);
            xs[k - 1]
        })
        .collect();
    let cell = |lo: f64, hi: f64| -> (usize, usize) {
        let a = xs.partition_point(|&x| x < lo);
        let b = xs.partition_point(|&x| x < hi);
        (a, b)
    };
    for _ in 0..500 {
        let mut max_change = 0.0f64;
        let mut next = centroids.clone();
        for i in 0..levels {
            let lo = if i == 0 {
                f64::NEG_INFINITY
            } else {
                (centroids[i - 1] + centroids[i]) / 2.0
            };
            let hi = if i == levels - 1 {
                f64::INFINITY
            } else {
                (centroids[i] + centroids[i + 1]) / 2.0
            };
            let (a, b) = cell(lo, hi);
            let w = cum_w[b] - cum_w[a];
            if w > 0.0 {
                next[i] = (cum_wx[b] - cum_wx[a]) / w;
            }
            max_change = max_change.max((next[i] - centroids[i]).abs());
        }
        centroids = next;
        if max_change < 1e-7 * sigma {
            break;
        }
    }
    // Symmetrise (the density is even) so encoding is sign-symmetric.
    for i in 0..levels / 2 {
        let m = (centroids[levels - 1 - i] - centroids[i]) / 2.0;
        centroids[i] = -m;
        centroids[levels - 1 - i] = m;
    }
    let c32: Vec<f32> = centroids.iter().map(|&c| c as f32).collect();
    let b32 = c32.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
    (b32, c32)
}

/// A query prepared for scoring against codes: the rotated, zero-padded query
/// and its squared norm.
#[derive(Debug, Clone)]
pub struct PreparedQuery {
    rotated: Vec<f32>,
    norm_sq: f32,
}

impl PreparedQuery {
    /// A query already rotated by this codec's rotation (or an equivalent
    /// orthogonal transform) and zero-padded to the padded dimension.
    #[allow(dead_code)]
    pub fn from_rotated(rotated: Vec<f32>, norm_sq: f32) -> Self {
        Self { rotated, norm_sq }
    }
    /// The rotated, zero-padded query.
    #[allow(dead_code)]
    pub fn into_rotated(self) -> Vec<f32> {
        self.rotated
    }
    /// Squared L2 norm of the query.
    pub fn norm_sq(&self) -> f32 {
        self.norm_sq
    }
}

/// Encoder/scorer for one (dimension, bit width).
#[derive(Debug)]
pub struct QuantCodec {
    dim: usize,
    padded: usize,
    bits: u32,
    rotation: Rotation,
    boundaries: Vec<f32>,
    centroids: Vec<f32>,
    /// `decode[byte * per_byte + t]` = centroid of sub-coordinate `t` of `byte`
    /// (only for bit widths dividing 8).
    decode: Vec<f32>,
}

fn codec_cache() -> &'static Mutex<HashMap<(u32, usize), Arc<QuantCodec>>> {
    static CACHE: OnceLock<Mutex<HashMap<(u32, usize), Arc<QuantCodec>>>> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

impl QuantCodec {
    /// Shared codec for `quant` over `dim`-dimensional vectors; `None` for
    /// [`VecQuant::None`].
    pub fn for_quant(quant: VecQuant, dim: usize) -> Option<Arc<QuantCodec>> {
        let bits = quant.bits()?;
        Some(Self::shared(bits, dim))
    }

    /// Shared codec for any bit width in `1..=8`.
    pub fn shared(bits: u32, dim: usize) -> Arc<QuantCodec> {
        let key = (bits, dim);
        let mut cache = codec_cache().lock().unwrap_or_else(|e| e.into_inner());
        cache
            .entry(key)
            .or_insert_with(|| Arc::new(QuantCodec::new(bits, dim)))
            .clone()
    }

    /// Build a codec (prefer [`Self::shared`]).
    pub fn new(bits: u32, dim: usize) -> Self {
        assert!((1..=8).contains(&bits), "bits must be in 1..=8");
        let padded = padded_dim(dim);
        let (boundaries, centroids) = lloyd_max(bits, padded);
        let mut decode = vec![];
        if 8 % bits == 0 {
            let per = (8 / bits) as usize;
            let mask = (1u32 << bits) - 1;
            decode = vec![0.0f32; 256 * per];
            for byte in 0..256u32 {
                for t in 0..per {
                    let idx = (byte >> (t as u32 * bits)) & mask;
                    decode[byte as usize * per + t] = centroids[idx as usize];
                }
            }
        }
        Self {
            dim,
            padded,
            bits,
            rotation: Rotation::new(padded),
            boundaries,
            centroids,
            decode,
        }
    }

    /// Input dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bit width.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Bytes of one code (header included).
    pub fn code_len(&self) -> usize {
        CODE_HEADER + (self.padded * self.bits as usize).div_ceil(8)
    }

    /// Rotate a vector of length `dim` into a zero-padded buffer.
    fn rotate(&self, v: &[f32], scale: f32) -> Vec<f32> {
        let mut row = vec![0.0f32; self.padded];
        for (r, x) in row.iter_mut().zip(v) {
            *r = *x * scale;
        }
        let mut scratch = vec![0.0f32; self.padded];
        self.rotation.apply(&mut row, &mut scratch);
        row
    }

    /// Encode `v` (length `dim`).
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        debug_assert_eq!(v.len(), self.dim);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return vec![0u8; self.code_len()];
        }
        let y = self.rotate(v, 1.0 / norm);
        self.encode_rotated_unit(&y, norm)
    }

    /// Encode an already rotated, zero-padded unit vector `y` whose original
    /// norm was `norm`.
    pub fn encode_rotated_unit(&self, y: &[f32], norm: f32) -> Vec<u8> {
        let mut code = vec![0u8; self.code_len()];
        let mut y_dot_hat = 0.0f32;
        let bits = self.bits as usize;
        let packed = &mut code[CODE_HEADER..];
        for (i, &yi) in y.iter().enumerate() {
            let idx = self.boundaries.partition_point(|&b| b < yi);
            y_dot_hat += yi * self.centroids[idx];
            let bit = i * bits;
            let mut val = idx as u32;
            let mut pos = bit;
            let mut remaining = bits;
            while remaining > 0 {
                let byte = pos / 8;
                let off = pos % 8;
                let take = remaining.min(8 - off);
                packed[byte] |= ((val & ((1 << take) - 1)) << off) as u8;
                val >>= take;
                pos += take;
                remaining -= take;
            }
        }
        let renorm = if y_dot_hat > 0.0 { norm / y_dot_hat } else { 0.0 };
        code[0..4].copy_from_slice(&renorm.to_le_bytes());
        code[4..8].copy_from_slice(&norm.to_le_bytes());
        code
    }

    /// Prepare a query (length `dim`) for scoring.
    pub fn prepare(&self, q: &[f32]) -> PreparedQuery {
        PreparedQuery {
            rotated: self.rotate(q, 1.0),
            norm_sq: q.iter().map(|x| x * x).sum(),
        }
    }

    /// `||v||` stored in a code.
    pub fn code_norm(code: &[u8]) -> f32 {
        f32::from_le_bytes([code[4], code[5], code[6], code[7]])
    }

    /// Estimate of `<q, v>` from `v`'s code.
    pub fn dot(&self, q: &PreparedQuery, code: &[u8]) -> f32 {
        let renorm = f32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        if renorm == 0.0 {
            return 0.0;
        }
        let packed = &code[CODE_HEADER..];
        let raw = match self.bits {
            1 => dot_table::<8, 2>(&q.rotated, packed, &self.decode),
            2 => dot_table::<4, 4>(&q.rotated, packed, &self.decode),
            4 => dot_table::<2, 8>(&q.rotated, packed, &self.decode),
            8 => dot_table::<1, 16>(&q.rotated, packed, &self.decode),
            _ => self.dot_generic(&q.rotated, packed),
        };
        raw * renorm
    }

    fn dot_generic(&self, qr: &[f32], packed: &[u8]) -> f32 {
        let bits = self.bits as usize;
        let mut acc = 0.0f32;
        for (i, q) in qr.iter().enumerate() {
            let mut pos = i * bits;
            let mut idx = 0u32;
            let mut got = 0;
            while got < bits {
                let byte = pos / 8;
                let off = pos % 8;
                let take = (bits - got).min(8 - off);
                idx |= ((packed[byte] as u32 >> off) & ((1 << take) - 1)) << got;
                got += take;
                pos += take;
            }
            acc += q * self.centroids[idx as usize];
        }
        acc
    }
}

/// `Σ q_i ĉ_i` with codes decoded through a per-byte table. `PER` coordinates
/// per byte, `BYTES` bytes per step (`PER * BYTES` = 16 accumulator lanes).
#[inline]
fn dot_table<const PER: usize, const BYTES: usize>(qr: &[f32], packed: &[u8], dec: &[f32]) -> f32 {
    assert_eq!(dec.len(), 256 * PER);
    assert_eq!(PER * BYTES, 16);
    // SAFETY: `dec` holds exactly 256 * PER contiguous f32s (checked above),
    // which is the layout of `[[f32; PER]; 256]`.
    let dec: &[[f32; PER]; 256] = unsafe { &*(dec.as_ptr() as *const [[f32; PER]; 256]) };
    let mut acc = [0.0f32; 16];
    for (qs, bs) in qr.chunks_exact(16).zip(packed.chunks_exact(BYTES)) {
        let qs: &[f32; 16] = qs.try_into().unwrap();
        for j in 0..BYTES {
            let d = &dec[bs[j] as usize];
            for t in 0..PER {
                acc[j * PER + t] += qs[j * PER + t] * d[t];
            }
        }
    }
    acc.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut rng = SplitMix64(seed);
        (0..dim)
            .map(|_| (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            .collect()
    }

    #[test]
    fn rotation_preserves_norm_and_dot() {
        let r = Rotation::new(192);
        let a = rand_vec(1, 192);
        let b = rand_vec(2, 192);
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        let (mut ra, mut rb, mut s) = (a.clone(), b.clone(), vec![0.0; 192]);
        r.apply(&mut ra, &mut s);
        r.apply(&mut rb, &mut s);
        assert!((dot(&a, &a) - dot(&ra, &ra)).abs() < 1e-3);
        assert!((dot(&a, &b) - dot(&ra, &rb)).abs() < 1e-3);
    }

    #[test]
    fn codebook_is_symmetric_and_ascending() {
        for bits in [1, 2, 3, 4, 8] {
            let (b, c) = lloyd_max(bits, 1536);
            assert_eq!(c.len(), 1 << bits);
            assert_eq!(b.len(), (1 << bits) - 1);
            assert!(c.windows(2).all(|w| w[0] < w[1]));
            assert!((c[0] + c[c.len() - 1]).abs() < 1e-6);
        }
        // 1-bit Lloyd-Max for a Gaussian is ±sqrt(2/π)·σ.
        let (_, c) = lloyd_max(1, 1536);
        let expect = (2.0 / std::f64::consts::PI).sqrt() / 1536f64.sqrt();
        assert!((c[1] as f64 - expect).abs() / expect < 0.01, "{c:?} {expect}");
    }

    #[test]
    fn self_dot_is_exact_and_estimates_are_close() {
        for bits in [1, 2, 3, 4, 8] {
            let codec = QuantCodec::new(bits, 300);
            let v = rand_vec(7, 300);
            let code = codec.encode(&v);
            assert_eq!(code.len(), codec.code_len());
            let q = codec.prepare(&v);
            let est = codec.dot(&q, &code);
            assert!((est - q.norm_sq()).abs() / q.norm_sq() < 1e-3, "bits {bits}");
            // A different vector: the error shrinks with the bit width.
            let w = rand_vec(8, 300);
            let exact: f32 = v.iter().zip(&w).map(|(a, b)| a * b).sum();
            let est = codec.dot(&codec.prepare(&w), &code);
            let scale = q.norm_sq().sqrt() * w.iter().map(|x| x * x).sum::<f32>().sqrt();
            let tol = match bits {
                1 => 0.25,
                2 => 0.12,
                3 => 0.08,
                4 => 0.05,
                _ => 0.01,
            };
            assert!((est - exact).abs() / scale < tol, "bits {bits}: {est} vs {exact}");
        }
    }

    #[test]
    fn zero_vector_encodes_to_zero_estimate() {
        let codec = QuantCodec::new(4, 64);
        let code = codec.encode(&[0.0; 64]);
        assert_eq!(QuantCodec::code_norm(&code), 0.0);
        assert_eq!(codec.dot(&codec.prepare(&[1.0; 64]), &code), 0.0);
    }

    #[test]
    fn packed_table_kernel_matches_generic() {
        for bits in [1, 2, 4, 8] {
            let codec = QuantCodec::new(bits, 256);
            let v = rand_vec(3, 256);
            let code = codec.encode(&v);
            let q = codec.prepare(&rand_vec(4, 256));
            let fast = codec.dot(&q, &code);
            let renorm = f32::from_le_bytes(code[0..4].try_into().unwrap());
            let slow = codec.dot_generic(&q.rotated, &code[CODE_HEADER..]) * renorm;
            assert!((fast - slow).abs() < 1e-4 * (1.0 + slow.abs()), "bits {bits}");
        }
    }
}
