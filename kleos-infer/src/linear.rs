/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Weight loading at a chosen precision: dense float or GGML block-quantised.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use serde::Deserialize;

use crate::{Error, Precision, Result};

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// A safetensors file read tensor by tensor with plain reads (no mmap), so
/// peak memory during load is the converted weights plus one source tensor
/// (or one row chunk of the token table), not the whole f32 checkpoint.
pub(crate) struct TensorFile {
    file: Mutex<File>,
    base: u64,
    entries: HashMap<String, Entry>,
}

impl TensorFile {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut n = [0u8; 8];
        file.read_exact(&mut n)?;
        let n = u64::from_le_bytes(n);
        let mut header = vec![0u8; n as usize];
        file.read_exact(&mut header)?;
        let mut raw: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)?;
        raw.remove("__metadata__");
        let entries = raw
            .into_iter()
            .map(|(k, v)| Ok((k, serde_json::from_value(v)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            file: Mutex::new(file),
            base: 8 + n,
            entries,
        })
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.entries
            .get(name)
            .ok_or_else(|| Error::Config(format!("tensor '{name}' not in checkpoint")))
    }

    /// Rows `[r0, r1)` of tensor `name` (all rows of a 1-d tensor), on the CPU.
    fn read_rows(&self, name: &str, r0: usize, r1: usize) -> Result<Tensor> {
        let e = self.entry(name)?;
        let dtype = match e.dtype.as_str() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            d => return Err(Error::Config(format!("unsupported dtype {d} for {name}"))),
        };
        let row_elems: usize = e.shape.iter().skip(1).product();
        let row_bytes = row_elems * dtype.size_in_bytes();
        let mut shape = e.shape.clone();
        shape[0] = r1 - r0;
        let mut buf = vec![0u8; (r1 - r0) * row_bytes];
        {
            let mut f = self.file.lock().unwrap();
            f.seek(SeekFrom::Start(
                self.base + e.data_offsets.0 as u64 + (r0 * row_bytes) as u64,
            ))?;
            f.read_exact(&mut buf)?;
        }
        Ok(Tensor::from_raw_buffer(&buf, dtype, &shape, &Device::Cpu)?)
    }

    fn rows(&self, name: &str) -> Result<usize> {
        Ok(self.entry(name)?.shape[0])
    }

    fn load(&self, name: &str) -> Result<Tensor> {
        let n = self.rows(name)?;
        self.read_rows(name, 0, n)
    }
}

/// Rows per chunk when converting a token table.
const TABLE_CHUNK_ROWS: usize = 16384;
/// Rows per chunk of a dense (float) token table: 1024 × 768 × 4 B = 3 MB.
const DENSE_TABLE_CHUNK_ROWS: usize = 1024;

pub(crate) struct Loader<'a> {
    pub(crate) st: &'a TensorFile,
    pub(crate) device: &'a Device,
    pub(crate) precision: Precision,
    /// Weight bytes as loaded, for reporting.
    pub(crate) bytes: usize,
}

impl Loader<'_> {
    /// Float dtype of non-quantised weights.
    pub(crate) fn weight_dtype(&self) -> DType {
        match self.precision {
            Precision::F16 => DType::F16,
            Precision::BF16 => DType::BF16,
            _ => DType::F32,
        }
    }

    /// Activation dtype. f16 keeps f32 activations: Gemma-3 residual
    /// activations overflow f16's range (measured: NaN embeddings), so f16
    /// is a weight-storage format only. bf16 has f32's range and runs natively.
    pub(crate) fn act_dtype(&self) -> DType {
        match self.precision {
            Precision::BF16 => DType::BF16,
            _ => DType::F32,
        }
    }

    /// A small tensor (norm weights etc.), always f32.
    pub(crate) fn f32(&mut self, name: &str) -> Result<Tensor> {
        let t = self
            .st
            .load(name)?
            .to_dtype(DType::F32)?
            .to_device(self.device)?;
        self.bytes += t.elem_count() * 4;
        Ok(t)
    }

    fn quant_dtype(&self, linear: bool) -> Option<GgmlDType> {
        match (self.precision, linear) {
            (Precision::Q8, _) | (Precision::Q4, false) => Some(GgmlDType::Q8_0),
            (Precision::Q4, true) => Some(GgmlDType::Q4_0),
            _ => None,
        }
    }

    /// A `[out, in]` linear weight.
    pub(crate) fn linear(&mut self, name: &str) -> Result<Linear> {
        let w = self.st.load(name)?;
        let n = w.elem_count();
        match self.quant_dtype(true) {
            Some(q) if w.dim(1)? % q.block_size() == 0 => {
                let qt = QTensor::quantize_onto(&w, q, self.device)?;
                self.bytes += qt.storage_size_in_bytes();
                Ok(Linear::Quant(QMatMul::from_qtensor(qt)?))
            }
            _ => {
                let dt = self.weight_dtype();
                self.bytes += n * dt.size_in_bytes();
                let w = w.to_dtype(dt)?.to_device(self.device)?;
                Ok(Linear::Dense(candle_nn::Linear::new(w, None)))
            }
        }
    }

    /// A `[vocab, hidden]` token table, converted in row chunks.
    pub(crate) fn embedding(&mut self, name: &str) -> Result<Embedding> {
        let rows = self.st.rows(name)?;
        let chunks = (0..rows).step_by(TABLE_CHUNK_ROWS).map(|r0| (r0, (r0 + TABLE_CHUNK_ROWS).min(rows)));
        let first = self.st.read_rows(name, 0, 1)?;
        let hidden = first.dim(1)?;
        match self.quant_dtype(false) {
            Some(q) if hidden % q.block_size() == 0 => {
                // quantised in small row chunks into an anonymous-mmap staging
                // area (see `staged_table`): no large transient is freed into
                // libmalloc's cache, which stays in the process footprint
                let _ = chunks;
                let row_bytes = hidden / q.block_size() * q.type_size();
                let mut stage = memmap2::MmapMut::map_anon((rows * row_bytes).max(1))?;
                for r0 in (0..rows).step_by(DENSE_TABLE_CHUNK_ROWS) {
                    let r1 = (r0 + DENSE_TABLE_CHUNK_ROWS).min(rows);
                    let c = self.st.read_rows(name, r0, r1)?;
                    let qt = QTensor::quantize(&c, q)?;
                    stage[r0 * row_bytes..r1 * row_bytes].copy_from_slice(&qt.data()?);
                }
                let qt = qtensor_from_ggml(q, &stage[..rows * row_bytes], vec![rows, hidden], self.device)?;
                self.bytes += rows * row_bytes;
                drop(stage);
                Ok(Embedding::Quant(QMatMul::from_qtensor(qt)?))
            }
            _ => {
                let dt = self.weight_dtype();
                let w = match dt {
                    DType::F16 => self.staged_table::<half::f16>(name, rows, hidden, dt)?,
                    DType::BF16 => self.staged_table::<half::bf16>(name, rows, hidden, dt)?,
                    _ => self.staged_table::<f32>(name, rows, hidden, dt)?,
                };
                self.bytes += w.elem_count() * dt.size_in_bytes();
                Ok(Embedding::Dense(w))
            }
        }
    }
}

impl Loader<'_> {
    /// A dense `[rows, hidden]` table converted in small row chunks into an
    /// anonymous-mmap staging area, then copied once into its final storage
    /// (an exact-size Metal buffer, or the CPU tensor). The staging pages go
    /// straight back to the OS on unmap, so no full-size transient lands in
    /// libmalloc's large-block cache (which macOS keeps in the process
    /// footprint), and no power-of-two pooled buffer is used on Metal
    /// (`Tensor::cat` on the device rounded 77 MB up to 128 MB).
    fn staged_table<T: candle_core::WithDType>(
        &mut self,
        name: &str,
        rows: usize,
        hidden: usize,
        dt: DType,
    ) -> Result<Tensor> {
        let n = rows * hidden;
        let mut stage = memmap2::MmapMut::map_anon((n * std::mem::size_of::<T>()).max(1))?;
        // SAFETY: an anonymous map is page aligned and `n * size_of::<T>()` long.
        let buf: &mut [T] = unsafe { std::slice::from_raw_parts_mut(stage.as_mut_ptr() as *mut T, n) };
        for r0 in (0..rows).step_by(DENSE_TABLE_CHUNK_ROWS) {
            let r1 = (r0 + DENSE_TABLE_CHUNK_ROWS).min(rows);
            let c = self.st.read_rows(name, r0, r1)?.to_dtype(dt)?.flatten_all()?.to_vec1::<T>()?;
            buf[r0 * hidden..r1 * hidden].copy_from_slice(&c);
        }
        let t = Tensor::from_slice(buf, (rows, hidden), self.device)?;
        drop(stage);
        Ok(t)
    }
}

pub(crate) enum Linear {
    Dense(candle_nn::Linear),
    Quant(QMatMul),
}

impl Linear {
    /// `x @ W^T`, returned in `x`'s dtype.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Linear::Dense(l) => {
                let wdt = l.weight().dtype();
                if x.dtype() == wdt {
                    l.forward(x)?
                } else {
                    l.forward(&x.to_dtype(wdt)?)?.to_dtype(x.dtype())?
                }
            }
            Linear::Quant(q) => {
                let dt = x.dtype();
                let y = q.forward(&x.to_dtype(DType::F32)?.contiguous()?)?;
                y.to_dtype(dt)?
            }
        })
    }
}

pub(crate) enum Embedding {
    Dense(Tensor),
    Quant(QMatMul),
}

impl Embedding {
    /// Rows for `ids` (`[b, l]` u32), as f32 `[b, l, hidden]`.
    pub(crate) fn lookup(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Embedding::Dense(w) => {
                let (b, l) = ids.dims2()?;
                let h = w.dim(1)?;
                w.index_select(&ids.flatten_all()?, 0)?
                    .reshape((b, l, h))?
                    .to_dtype(DType::F32)?
            }
            Embedding::Quant(q) => q.embedding(ids)?.to_dtype(DType::F32)?,
        })
    }
}
