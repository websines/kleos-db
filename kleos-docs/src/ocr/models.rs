//! The Indra ONNX graphs on tract (pure Rust; no ONNX Runtime).
//!
//! The recognizers, the script classifier, the orientation model and the
//! order model optimize with symbolic batch and width.  The detector's graph
//! does not (a reshape of a symbolic half size), so it is optimized once per
//! input shape and cached: pages scale to a fixed long side, so a document
//! has few shapes.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use tract_onnx::prelude::*;

pub type Plan = TypedSimplePlan;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model {0}: {1}")]
    Load(String, String),
    #[error("run {0}: {1}")]
    Run(String, String),
}

fn load_symbolic(path: &Path, keep_outputs: Option<usize>) -> Result<Arc<Plan>, ModelError> {
    let name = path.display().to_string();
    let m = tract_onnx::onnx()
        .model_for_path(path)
        .and_then(|mut m| {
            // keep only the first outputs: the graph that only feeds the
            // others (the recognizers' top-k for the LM) is pruned away
            if let Some(n) = keep_outputs {
                let outs = m.output_outlets()?.to_vec();
                if n < outs.len() {
                    m.select_output_outlets(&outs[..n])?;
                }
            }
            Ok(m)
        })
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable())
        .map_err(|e| ModelError::Load(name, format!("{e:?}")))?;
    Ok(m)
}

/// A graph optimized per concrete input shape (the detector).
pub struct ShapedModel {
    proto: InferenceModel,
    name: String,
    plans: Mutex<HashMap<Vec<usize>, Arc<Plan>>>,
}

impl ShapedModel {
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let name = path.display().to_string();
        let proto = tract_onnx::onnx().model_for_path(path).map_err(|e| ModelError::Load(name.clone(), format!("{e:?}")))?;
        Ok(ShapedModel { proto, name, plans: Mutex::new(HashMap::new()) })
    }

    fn plan(&self, shape: &[usize]) -> Result<Arc<Plan>, ModelError> {
        if let Some(p) = self.plans.lock().unwrap().get(shape) {
            return Ok(p.clone());
        }
        let p = self
            .proto
            .clone()
            .with_input_fact(0, f32::fact(shape).into())
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| ModelError::Load(self.name.clone(), format!("{e:?}")))?;
        self.plans.lock().unwrap().insert(shape.to_vec(), p.clone());
        Ok(p)
    }

    /// One f32 input of `shape`; every output as (shape, data).
    pub fn run(&self, shape: &[usize], data: Vec<f32>) -> Result<Vec<(Vec<usize>, Vec<f32>)>, ModelError> {
        let p = self.plan(shape)?;
        let t = Tensor::from_shape(shape, &data).map_err(|e| ModelError::Run(self.name.clone(), format!("{e:?}")))?;
        let out = p.run(tvec!(t.into())).map_err(|e| ModelError::Run(self.name.clone(), format!("{e:?}")))?;
        out.iter()
            .map(|o| {
                let v = o.cast_to::<f32>().map_err(|e| ModelError::Run(self.name.clone(), format!("{e:?}")))?;
                Ok((o.shape().to_vec(), v.try_as_plain_ram().and_then(|p| p.as_slice::<f32>().map(|s| s.to_vec())).map_err(|e| ModelError::Run(self.name.clone(), format!("{e:?}")))?))
            })
            .collect()
    }
}

/// A graph with symbolic dimensions, run on any input shape.
pub struct Model {
    plan: Arc<Plan>,
    name: String,
    pub outputs: Vec<String>,
}

/// An input tensor for `Model::run`.
pub enum Input {
    F32(Vec<usize>, Vec<f32>),
    I64(Vec<usize>, Vec<i64>),
    Bool(Vec<usize>, Vec<bool>),
}

/// An output tensor.
#[derive(Debug)]
pub enum Output {
    F32(Vec<usize>, Vec<f32>),
    I64(Vec<usize>, Vec<i64>),
}

impl Output {
    pub fn shape(&self) -> &[usize] {
        match self {
            Output::F32(s, _) | Output::I64(s, _) => s,
        }
    }
    pub fn f32(&self) -> Vec<f32> {
        match self {
            Output::F32(_, v) => v.clone(),
            Output::I64(_, v) => v.iter().map(|&x| x as f32).collect(),
        }
    }
    pub fn i64(&self) -> Vec<i64> {
        match self {
            Output::I64(_, v) => v.clone(),
            Output::F32(_, v) => v.iter().map(|&x| x as i64).collect(),
        }
    }
}

impl Model {
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        Self::load_outputs(path, None)
    }

    /// Only the first `n` outputs (None: all).
    pub fn load_outputs(path: &Path, n: Option<usize>) -> Result<Self, ModelError> {
        let plan = load_symbolic(path, n)?;
        let outputs = plan
            .model()
            .outputs
            .iter()
            .map(|o| plan.model().outlet_label(*o).map(|s| s.to_string()).unwrap_or_else(|| plan.model().node(o.node).name.clone()))
            .collect();
        Ok(Model { plan, name: path.display().to_string(), outputs })
    }

    pub fn run(&self, inputs: Vec<Input>) -> Result<Vec<Output>, ModelError> {
        let err = |e: TractError| ModelError::Run(self.name.clone(), format!("{e:?}"));
        let mut tv: TVec<TValue> = tvec![];
        for i in inputs {
            let t = match i {
                Input::F32(s, d) => Tensor::from_shape(&s, &d),
                Input::I64(s, d) => Tensor::from_shape(&s, &d),
                Input::Bool(s, d) => Tensor::from_shape(&s, &d),
            }
            .map_err(err)?;
            tv.push(t.into());
        }
        let out = self.plan.run(tv).map_err(err)?;
        out.iter()
            .map(|o| {
                let shape = o.shape().to_vec();
                if o.datum_type() == i64::datum_type() || o.datum_type() == i32::datum_type() {
                    let v = o.cast_to::<i64>().map_err(err)?;
                    Ok(Output::I64(shape, v.try_as_plain_ram().and_then(|p| p.as_slice::<i64>().map(|s| s.to_vec())).map_err(err)?))
                } else {
                    let v = o.cast_to::<f32>().map_err(err)?;
                    Ok(Output::F32(shape, v.try_as_plain_ram().and_then(|p| p.as_slice::<f32>().map(|s| s.to_vec())).map_err(|e| ModelError::Run(self.name.clone(), format!("{e:?}")))?))
                }
            })
            .collect()
    }
}

/// A recognizer: the hand-written CPU path ([`super::fastnn::RecNet`]) or,
/// when that refuses the graph or `KLEOS_OCR_TRACT=1`, tract.
pub enum Net {
    Tract(Model),
    Native { net: Arc<super::fastnn::RecNet>, full: bool },
}

/// Whether tract is forced for the recognizers (the reference path).
pub fn tract_forced() -> bool {
    std::env::var("KLEOS_OCR_TRACT").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
}

impl Net {
    /// The native recognizer for `path`, or None (tract forced, or a graph it
    /// does not implement; the reason goes to stderr once).
    pub fn native(path: &Path) -> Option<Arc<super::fastnn::RecNet>> {
        if tract_forced() {
            return None;
        }
        match super::fastnn::RecNet::load(path) {
            Ok(n) => Some(Arc::new(n)),
            Err(e) => {
                eprintln!("kleos-docs: {} on tract ({e})", path.display());
                None
            }
        }
    }

    /// `native` when given, else tract with the first `outputs` outputs.
    pub fn load(path: &Path, native: Option<&Arc<super::fastnn::RecNet>>, outputs: Option<usize>) -> Result<Self, ModelError> {
        match native {
            Some(n) => Ok(Net::Native { net: n.clone(), full: outputs.is_none() }),
            None => Ok(Net::Tract(Model::load_outputs(path, outputs)?)),
        }
    }

    pub fn run(&self, inputs: Vec<Input>) -> Result<Vec<Output>, ModelError> {
        match self {
            Net::Tract(m) => m.run(inputs),
            Net::Native { net, full } => {
                let mut it = inputs.into_iter();
                match (it.next(), it.next()) {
                    (Some(Input::F32(s, x)), Some(Input::I64(_, w))) if s.len() == 4 && s[1] == 1 => net.run(s[0], s[2], s[3], &x, &w, *full),
                    _ => Err(ModelError::Run("rec (native)".into(), "expected x [b,1,h,w] and widths [b]".into())),
                }
            }
        }
    }
}

/// A 2-D net (the detector, the orientation model): the hand-written
/// interpreter ([`super::convnet::ConvNet`]) or, when that refuses the graph
/// or `KLEOS_OCR_TRACT=1`, tract optimized per input shape.
pub enum Net2D {
    Tract(ShapedModel),
    Native(super::convnet::ConvNet),
}

impl Net2D {
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        if !tract_forced() {
            match super::convnet::ConvNet::load(path) {
                Ok(n) => return Ok(Net2D::Native(n)),
                Err(e) => eprintln!("kleos-docs: {} on tract ({e})", path.display()),
            }
        }
        Ok(Net2D::Tract(ShapedModel::load(path)?))
    }

    /// One f32 input of `shape`; every output as (shape, data).
    pub fn run(&self, shape: &[usize], data: Vec<f32>) -> Result<Vec<(Vec<usize>, Vec<f32>)>, ModelError> {
        match self {
            Net2D::Tract(m) => m.run(shape, data),
            Net2D::Native(n) => n.run(shape, data),
        }
    }
}
