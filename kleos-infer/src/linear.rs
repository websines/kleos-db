/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Weight loading at a chosen precision: dense float or GGML block-quantised.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Module, Tensor};

use crate::{Precision, Result};

pub(crate) struct Loader<'a> {
    pub(crate) st: &'a MmapedSafetensors,
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
            .load(name, &Device::Cpu)?
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
        let w = self.st.load(name, &Device::Cpu)?;
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

    /// A `[vocab, hidden]` token table.
    pub(crate) fn embedding(&mut self, name: &str) -> Result<Embedding> {
        let w = self.st.load(name, &Device::Cpu)?;
        match self.quant_dtype(false) {
            Some(q) if w.dim(1)? % q.block_size() == 0 => {
                let qt = QTensor::quantize_onto(&w, q, self.device)?;
                self.bytes += qt.storage_size_in_bytes();
                Ok(Embedding::Quant(QMatMul::from_qtensor(qt)?))
            }
            _ => {
                let dt = self.weight_dtype();
                self.bytes += w.elem_count() * dt.size_in_bytes();
                Ok(Embedding::Dense(w.to_dtype(dt)?.to_device(self.device)?))
            }
        }
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
