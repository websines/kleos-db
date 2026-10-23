/*
 * Copyright 2022, The Cozo Project Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, RwLock};
#[cfg(feature = "columnar-io")]
use std::time::Duration;

use miette::{IntoDiagnostic, Report, Result};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyList, PyString, PyTuple};
use serde_json::json;

use ::kleos::*;

fn py_to_rows(ob: &PyAny) -> PyResult<Vec<Vec<DataValue>>> {
    let rows = ob.extract::<Vec<Vec<&PyAny>>>()?;
    let res: Vec<Vec<DataValue>> = rows
        .into_iter()
        .map(|row| row.into_iter().map(py_to_value).collect::<PyResult<_>>())
        .collect::<PyResult<_>>()?;
    Ok(res)
}

fn report2py(r: Report) -> PyErr {
    PyException::new_err(r.to_string())
}

fn py_to_named_rows(ob: &PyAny) -> PyResult<NamedRows> {
    let d = ob.downcast::<PyDict>()?;
    let rows = d
        .get_item("rows")?
        .ok_or_else(|| PyException::new_err("named rows must contain 'rows'"))?;
    let rows = py_to_rows(rows)?;
    let headers = d
        .get_item("headers")?
        .ok_or_else(|| PyException::new_err("named rows must contain 'headers'"))?;
    let headers = headers.extract::<Vec<String>>()?;
    Ok(NamedRows::new(headers, rows))
}

fn py_to_value(ob: &PyAny) -> PyResult<DataValue> {
    Ok(if ob.is_none() {
        DataValue::Null
    } else if let Ok(b) = ob.downcast::<PyBool>() {
        DataValue::from(b.is_true())
    } else if let Ok(i) = ob.extract::<i64>() {
        DataValue::from(i)
    } else if let Ok(f) = ob.extract::<f64>() {
        DataValue::from(f)
    } else if let Ok(s) = ob.extract::<String>() {
        DataValue::from(s)
    } else if let Ok(b) = ob.downcast::<PyBytes>() {
        DataValue::Bytes(b.as_bytes().to_vec())
    } else if let Ok(b) = ob.downcast::<PyByteArray>() {
        DataValue::Bytes(unsafe { b.as_bytes() }.to_vec())
    } else if let Ok(l) = ob.downcast::<PyTuple>() {
        let mut coll = Vec::with_capacity(l.len());
        for el in l {
            let el = py_to_value(el)?;
            coll.push(el)
        }
        DataValue::List(coll)
    } else if let Ok(l) = ob.downcast::<PyList>() {
        let mut coll = Vec::with_capacity(l.len());
        for el in l {
            let el = py_to_value(el)?;
            coll.push(el)
        }
        DataValue::List(coll)
    } else if let Ok(d) = ob.downcast::<PyDict>() {
        let mut coll = serde_json::Map::default();
        for (k, v) in d {
            let k = serde_json::Value::from(py_to_value(k)?);
            let k = match k {
                serde_json::Value::String(s) => s,
                s => s.to_string(),
            };
            let v = serde_json::Value::from(py_to_value(v)?);
            coll.insert(k, v);
        }
        DataValue::Json(JsonData(json!(coll)))
    } else {
        return Err(PyException::new_err(format!(
            "Cannot convert {ob} into Cozo value"
        )));
    })
}

fn convert_params(ob: &PyDict) -> PyResult<BTreeMap<String, DataValue>> {
    let mut ret = BTreeMap::new();
    for (k, v) in ob {
        let k: String = k.extract()?;
        let v = py_to_value(v)?;
        ret.insert(k, v);
    }
    Ok(ret)
}

/// Build a [`HybridSearch`] from a Python dict, applying the Rust `Default` for
/// any omitted field (kleos fork). Mirrors the field names of the Rust struct.
#[cfg(feature = "cypher")]
fn cy_req_str(d: &PyDict, k: &str) -> PyResult<String> {
    d.get_item(k)?
        .ok_or_else(|| PyException::new_err(format!("schema entry needs '{k}'")))?
        .extract()
}

#[cfg(feature = "cypher")]
fn cy_opt_str(d: &PyDict, k: &str) -> PyResult<Option<String>> {
    Ok(match d.get_item(k)? {
        Some(x) if !x.is_none() => Some(x.extract()?),
        _ => None,
    })
}

#[cfg(feature = "cypher")]
fn cy_opt_val(d: &PyDict, k: &str) -> PyResult<Option<DataValue>> {
    Ok(match d.get_item(k)? {
        Some(x) if !x.is_none() => Some(py_to_value(x)?),
        _ => None,
    })
}

/// Build a `CypherGraphSchema` from a Python dict mirroring the Rust fields:
/// `{"nodes": [{label, relation, id_col, label_col?, label_value?, filter?}, ...],
///   "edges": [{rel_type, relation, from_col, to_col, type_col?, type_value?, eid_col?, filter?}, ...]}`.
#[cfg(feature = "cypher")]
fn py_to_cypher_schema(d: &PyDict) -> PyResult<CypherGraphSchema> {
    let mut schema = CypherGraphSchema::default();
    if let Some(v) = d.get_item("nodes")? {
        for it in v.downcast::<PyList>()? {
            let nd = it.downcast::<PyDict>()?;
            schema.nodes.push(NodeMap {
                label: cy_req_str(nd, "label")?,
                relation: cy_req_str(nd, "relation")?,
                id_col: cy_req_str(nd, "id_col")?,
                label_col: cy_opt_str(nd, "label_col")?,
                label_value: cy_opt_val(nd, "label_value")?,
                filter: cy_opt_str(nd, "filter")?,
            });
        }
    }
    if let Some(v) = d.get_item("edges")? {
        for it in v.downcast::<PyList>()? {
            let ed = it.downcast::<PyDict>()?;
            schema.edges.push(EdgeMap {
                rel_type: cy_req_str(ed, "rel_type")?,
                relation: cy_req_str(ed, "relation")?,
                from_col: cy_req_str(ed, "from_col")?,
                to_col: cy_req_str(ed, "to_col")?,
                type_col: cy_opt_str(ed, "type_col")?,
                type_value: cy_opt_val(ed, "type_value")?,
                eid_col: cy_opt_str(ed, "eid_col")?,
                filter: cy_opt_str(ed, "filter")?,
            });
        }
    }
    Ok(schema)
}

/// A normalization spec: `"minmax"` | `"zscore"` | `"none"` |
/// `"neg_cosine_distance"` | `"cosine_similarity"` | `"bm25"` |
/// `["range", lo, hi]` | `["tmm", lo]`.
fn py_to_score_norm(v: &PyAny) -> PyResult<ScoreNorm> {
    if let Ok(s) = v.extract::<String>() {
        return match s.as_str() {
            "minmax" => Ok(ScoreNorm::MinMax),
            "zscore" => Ok(ScoreNorm::ZScore),
            "none" => Ok(ScoreNorm::Raw),
            "neg_cosine_distance" => Ok(ScoreNorm::NegCosineDistance),
            "cosine_similarity" => Ok(ScoreNorm::CosineSimilarity),
            "bm25" => Ok(ScoreNorm::Bm25),
            other => Err(PyException::new_err(format!(
                "hybrid_search: unknown normalization {other:?}"
            ))),
        };
    }
    let l: Vec<&PyAny> = v.extract()?;
    let kind: String = l
        .first()
        .ok_or_else(|| PyException::new_err("hybrid_search: empty normalization list"))?
        .extract()?;
    match (kind.as_str(), l.len()) {
        ("range", 3) => Ok(ScoreNorm::Range {
            lo: l[1].extract()?,
            hi: l[2].extract()?,
        }),
        ("tmm", 2) => Ok(ScoreNorm::Tmm { lo: l[1].extract()? }),
        _ => Err(PyException::new_err(
            "hybrid_search: a list normalization must be [\"range\", lo, hi] or [\"tmm\", lo]",
        )),
    }
}

/// `"rrf"` or `{"kind": "weighted", "weights": {leg: w}, "norm": spec,
/// "leg_norms": {leg: spec}}`.
fn py_to_fusion(v: &PyAny) -> PyResult<Fusion> {
    if let Ok(s) = v.extract::<String>() {
        return match s.as_str() {
            "rrf" => Ok(Fusion::Rrf),
            "weighted" => Ok(Fusion::Weighted(WeightedFusionParams::default())),
            other => Err(PyException::new_err(format!(
                "hybrid_search: unknown fusion {other:?} (expected \"rrf\" or \"weighted\")"
            ))),
        };
    }
    let d = v.downcast::<PyDict>()?;
    let kind: String = match d.get_item("kind")? {
        Some(k) => k.extract()?,
        None => "weighted".into(),
    };
    if kind == "rrf" {
        return Ok(Fusion::Rrf);
    }
    if kind != "weighted" {
        return Err(PyException::new_err(format!(
            "hybrid_search: unknown fusion kind {kind:?}"
        )));
    }
    let mut p = WeightedFusionParams::default();
    if let Some(w) = d.get_item("weights")? {
        let w = w.downcast::<PyDict>()?;
        for (k, x) in w.iter() {
            p.weights.push((k.extract()?, x.extract()?));
        }
    }
    if let Some(n) = d.get_item("norm")? {
        p.norm = py_to_score_norm(n)?;
    }
    if let Some(ln) = d.get_item("leg_norms")? {
        let ln = ln.downcast::<PyDict>()?;
        for (k, x) in ln.iter() {
            p.leg_norms.push((k.extract()?, py_to_score_norm(x)?));
        }
    }
    Ok(Fusion::Weighted(p))
}

/// A Python callable `(query: str, docs: list[str]) -> list[float]` as a
/// kleos `Reranker`.
struct PyReranker(Py<PyAny>);

impl Reranker for PyReranker {
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        Python::with_gil(|py| -> Result<Vec<f32>> {
            let args = PyTuple::new(
                py,
                vec![
                    PyString::new(py, query).to_object(py),
                    PyList::new(py, docs).to_object(py),
                ],
            );
            let res = self.0.as_ref(py).call1(args).into_diagnostic()?;
            res.extract::<Vec<f32>>().into_diagnostic()
        })
    }
}

/// A Python callable `(texts: list[str]) -> list[list[float]]` as a kleos
/// `Embedder` (for `embed('name', text)` and computed columns).
struct PyEmbedder {
    callback: Py<PyAny>,
    dim: usize,
    max_batch: usize,
    model_id: Option<String>,
}

impl ::kleos::embed::Embedder for PyEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Python::with_gil(|py| -> Result<Vec<Vec<f32>>> {
            let res = self
                .callback
                .as_ref(py)
                .call1((PyList::new(py, texts),))
                .map_err(|e| miette::miette!("Python embedder raised: {e}"))?;
            res.extract::<Vec<Vec<f32>>>()
                .map_err(|e| miette::miette!("Python embedder must return list[list[float]]: {e}"))
        })
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn model_id(&self) -> Option<String> {
        self.model_id.clone()
    }
    fn max_batch(&self) -> usize {
        self.max_batch
    }
}

fn py_to_hybrid_search(d: &PyDict) -> PyResult<HybridSearch> {
    let mut q = HybridSearch::default();
    if let Some(v) = d.get_item("relation")? {
        q.relation = v.extract()?;
    }
    if let Some(v) = d.get_item("id_col")? {
        q.id_col = v.extract()?;
    }
    if let Some(v) = d.get_item("vector_index")? {
        q.vector_index = v.extract()?;
    }
    if let Some(v) = d.get_item("query_vector")? {
        q.query_vector = v.extract()?;
    }
    if let Some(v) = d.get_item("vector_f64")? {
        q.vector_f64 = v.extract()?;
    }
    if let Some(v) = d.get_item("vector_k")? {
        q.vector_k = v.extract()?;
    }
    if let Some(v) = d.get_item("ef")? {
        q.ef = v.extract()?;
    }
    if let Some(v) = d.get_item("fts_index")? {
        q.fts_index = v.extract()?;
    }
    if let Some(v) = d.get_item("query_text")? {
        q.query_text = v.extract()?;
    }
    if let Some(v) = d.get_item("fts_k")? {
        q.fts_k = v.extract()?;
    }
    if let Some(v) = d.get_item("rrf_k")? {
        q.rrf_k = v.extract()?;
    }
    if let Some(v) = d.get_item("limit")? {
        q.limit = v.extract()?;
    }
    if let Some(v) = d.get_item("detailed")? {
        q.detailed = v.extract()?;
    }
    if let Some(v) = d.get_item("extra_lists")? {
        let items = v.downcast::<PyList>()?;
        let mut lists = Vec::with_capacity(items.len());
        for it in items {
            let ld = it.downcast::<PyDict>()?;
            let label = ld
                .get_item("label")?
                .ok_or_else(|| PyException::new_err("extra_lists entry needs 'label'"))?
                .extract()?;
            let rule_body = ld
                .get_item("rule_body")?
                .ok_or_else(|| PyException::new_err("extra_lists entry needs 'rule_body'"))?
                .extract()?;
            lists.push(HybridList { label, rule_body });
        }
        q.extra_lists = lists;
    }
    if let Some(v) = d.get_item("graph_legs")? {
        let items = v.downcast::<PyList>()?;
        let mut legs = Vec::with_capacity(items.len());
        for it in items {
            let gd = it.downcast::<PyDict>()?;
            // Every KNOWN key is optional and defaults like the Rust struct;
            // the builder's validation owns the invariants (which fields each
            // mode requires) so the rules live in exactly one place. Unknown
            // keys are REJECTED: with all keys optional, a typo ('max_hop',
            // 'seed_from_leg', a trailing space) would otherwise silently run
            // the leg with defaults — plausible results, wrong depth/mode, no
            // signal anywhere.
            const KNOWN_LEG_KEYS: [&str; 15] = [
                "label",
                "edge_relation",
                "from_col",
                "to_col",
                "seeds",
                "max_hops",
                "undirected",
                "max_nodes",
                "max_cost",
                "weight_col",
                "graph",
                "seed_from_legs",
                "gate_relation",
                "gate_cols",
                "admit",
            ];
            for key in gd.keys() {
                let key: String = key.extract()?;
                if !KNOWN_LEG_KEYS.contains(&key.as_str()) {
                    return Err(PyException::new_err(format!(
                        "graph_legs entry has unknown key {key:?}; known keys: {}",
                        KNOWN_LEG_KEYS.join(", ")
                    )));
                }
            }
            let mut leg = GraphLeg::default();
            if let Some(x) = gd.get_item("label")? {
                leg.label = x.extract()?;
            }
            if let Some(x) = gd.get_item("edge_relation")? {
                leg.edge_relation = x.extract()?;
            }
            if let Some(x) = gd.get_item("from_col")? {
                leg.from_col = x.extract()?;
            }
            if let Some(x) = gd.get_item("to_col")? {
                leg.to_col = x.extract()?;
            }
            if let Some(x) = gd.get_item("seeds")? {
                let seeds_list = x.downcast::<PyList>()?;
                let mut seeds = Vec::with_capacity(seeds_list.len());
                for s in seeds_list {
                    seeds.push(py_to_value(s)?);
                }
                leg.seeds = seeds;
            }
            if let Some(x) = gd.get_item("max_hops")? {
                leg.max_hops = x.extract()?;
            }
            if let Some(x) = gd.get_item("undirected")? {
                leg.undirected = x.extract()?;
            }
            // Budgeted-expansion mode (0.13.0): presence of max_nodes
            // switches the leg; the rest configure it.
            if let Some(x) = gd.get_item("max_nodes")? {
                leg.max_nodes = x.extract()?;
            }
            if let Some(x) = gd.get_item("max_cost")? {
                leg.max_cost = x.extract()?;
            }
            if let Some(x) = gd.get_item("weight_col")? {
                leg.weight_col = x.extract()?;
            }
            if let Some(x) = gd.get_item("graph")? {
                leg.graph = x.extract()?;
            }
            if let Some(x) = gd.get_item("seed_from_legs")? {
                leg.seed_from_legs = x.extract()?;
            }
            if let Some(x) = gd.get_item("gate_relation")? {
                leg.gate_relation = x.extract()?;
            }
            if let Some(x) = gd.get_item("gate_cols")? {
                leg.gate_cols = x.extract()?;
            }
            if let Some(x) = gd.get_item("admit")? {
                leg.admit = x.extract()?;
            }
            legs.push(leg);
        }
        q.graph_legs = legs;
    }
    if let Some(v) = d.get_item("fusion")? {
        if !v.is_none() {
            q.fusion = py_to_fusion(v)?;
        }
    }
    if let Some(v) = d.get_item("rerank")? {
        if !v.is_none() {
            let rd = v.downcast::<PyDict>()?;
            let mut r = RerankParams::default();
            if let Some(x) = rd.get_item("model")? {
                r.model = x.extract()?;
            }
            if let Some(x) = rd.get_item("top_n")? {
                r.top_n = x.extract()?;
            }
            if let Some(x) = rd.get_item("text_field")? {
                r.text_field = x.extract()?;
            }
            if let Some(x) = rd.get_item("query")? {
                r.query = x.extract()?;
            }
            if let Some(x) = rd.get_item("batch_size")? {
                r.batch_size = x.extract()?;
            }
            q.rerank = Some(r);
        }
    }
    if let Some(v) = d.get_item("mmr")? {
        if !v.is_none() {
            let md = v.downcast::<PyDict>()?;
            let mut m = MmrParams::default();
            if let Some(x) = md.get_item("lambda")? {
                m.lambda = x.extract()?;
            }
            if let Some(x) = md.get_item("k")? {
                m.k = x.extract()?;
            }
            if let Some(x) = md.get_item("embedding_col")? {
                m.embedding_col = x.extract()?;
            }
            q.mmr = Some(m);
        }
    }
    Ok(q)
}

fn options_to_py(opts: BTreeMap<String, DataValue>, py: Python<'_>) -> PyResult<PyObject> {
    let ret = PyDict::new(py);

    for (k, v) in opts {
        let val = value_to_py(v, py);
        ret.set_item(k, val)?;
    }
    Ok(ret.into())
}

fn json_to_py(val: serde_json::Value, py: Python<'_>) -> PyObject {
    match val {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.into_py(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py(py)
            } else {
                py.None()
            }
        }
        serde_json::Value::String(s) => s.into_py(py),
        serde_json::Value::Array(a) => {
            let vs: Vec<_> = a.into_iter().map(|v| json_to_py(v, py)).collect();
            vs.into_py(py)
        }
        serde_json::Value::Object(o) => {
            let d = PyDict::new(py);
            for (k, v) in o {
                d.set_item(k, json_to_py(v, py)).unwrap();
            }
            d.into()
        }
    }
}

fn value_to_py(val: DataValue, py: Python<'_>) -> PyObject {
    match val {
        DataValue::Null => py.None(),
        DataValue::Bool(b) => b.into_py(py),
        DataValue::Num(num) => match num {
            Num::Int(i) => i.into_py(py),
            Num::Float(f) => f.into_py(py),
        },
        DataValue::Str(s) => s.as_str().into_py(py),
        DataValue::Bytes(b) => PyBytes::new(py, &b).into(),
        DataValue::Uuid(uuid) => uuid.0.to_string().into_py(py),
        DataValue::Regex(rx) => rx.0.as_str().into_py(py),
        DataValue::List(l) => {
            let vs: Vec<_> = l.into_iter().map(|v| value_to_py(v, py)).collect();
            vs.into_py(py)
        }
        DataValue::Set(l) => {
            let vs: Vec<_> = l.into_iter().map(|v| value_to_py(v, py)).collect();
            vs.into_py(py)
        }
        DataValue::Validity(vld) => {
            [vld.timestamp.0 .0.into_py(py), vld.is_assert.0.into_py(py)].into_py(py)
        }
        DataValue::Bot => py.None(),
        DataValue::Vec(v) => match v {
            Vector::F32(a) => {
                let vs: Vec<_> = a.into_iter().map(|v| v.into_py(py)).collect();
                vs.into_py(py)
            }
            Vector::F64(a) => {
                let vs: Vec<_> = a.into_iter().map(|v| v.into_py(py)).collect();
                vs.into_py(py)
            }
        },
        DataValue::Json(JsonData(j)) => json_to_py(j, py),
    }
}

fn rows_to_py_rows(rows: Vec<Vec<DataValue>>, py: Python<'_>) -> PyObject {
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|val| value_to_py(val, py))
                .collect::<Vec<_>>()
                .into_py(py)
        })
        .collect::<Vec<_>>()
        .into_py(py)
}

fn named_rows_to_py(named_rows: NamedRows, py: Python<'_>) -> PyObject {
    let rows = rows_to_py_rows(named_rows.rows, py);
    let headers = named_rows.headers.into_py(py);
    let next = match named_rows.next {
        None => py.None(),
        Some(nxt) => named_rows_to_py(*nxt, py),
    };
    BTreeMap::from([("rows", rows), ("headers", headers), ("next", next)]).into_py(py)
}

// Interior mutability so that `close()` can take `&self` (kleos fork, item D).
// PyO3's PyCell runtime borrow check would raise "Already borrowed" if `close`
// took `&mut self` while a concurrent `run_script(&self)` held a shared borrow
// across its GIL-released blocking call. An `RwLock` (NOT a `RefCell`, which is
// not `Sync` and cannot back a shareable pyclass) lets every method take `&self`:
// read paths clone the cheap Arc-backed `DbInstance` out of a *momentary* guard
// and drop the guard before the blocking engine call; `close` takes a write
// guard and `take()`s. Concurrent readers never block `close`, and `close` never
// waits out an in-flight query.
#[pyclass]
struct CozoDbPy {
    db: RwLock<Option<DbInstance>>,
    /// Ids of live subscriptions made through this handle, so `close` can end
    /// them. Without this a closed handle left every subscription running
    /// (its delivery thread, the engine's live worker and the `Db` it holds)
    /// with no way to stop it: `unsubscribe` on a closed handle is a no-op.
    subs: Mutex<BTreeSet<u64>>,
}

#[pyclass]
struct CozoDbMulTx {
    tx: MultiTransaction,
}

const DB_CLOSED_MSG: &str = r##"{"ok":false,"message":"database closed"}"##;

impl CozoDbPy {
    /// Run a memory-API op (kleos fork) with keyword options as JSON.
    fn memory_op(
        &self,
        py: Python<'_>,
        op: &str,
        extra: Vec<(&str, serde_json::Value)>,
        opts: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let mut args = serde_json::Map::new();
        if let Some(o) = opts {
            for (k, v) in o {
                let k: String = k.extract()?;
                args.insert(k, serde_json::Value::from(py_to_value(v)?));
            }
        }
        for (k, v) in extra {
            args.insert(k.to_string(), v);
        }
        let out = py
            .allow_threads(|| db.memory_call(op, serde_json::Value::Object(args)))
            .map_err(report2py)?;
        Ok(json_to_py(out, py))
    }
    /// Clone the Arc-backed [`DbInstance`] out of a momentary read guard and
    /// drop the guard before returning, so callers never hold the lock across
    /// the blocking engine call they run under `py.allow_threads`. Returns a
    /// clear Python error (never panics) if the handle has already been closed.
    fn db_ref(&self) -> PyResult<DbInstance> {
        self.db
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| PyException::new_err(DB_CLOSED_MSG))
    }
}

#[pymethods]
impl CozoDbPy {
    #[new]
    fn new(engine: &str, path: &str, options: &str) -> PyResult<Self> {
        match DbInstance::new(engine, path, options) {
            Ok(db) => Ok(Self {
                db: RwLock::new(Some(db)),
                subs: Mutex::new(BTreeSet::new()),
            }),
            Err(err) => Err(PyException::new_err(format!("{err:?}"))),
        }
    }
    /// Run a CozoScript query. `timeout` (kleos fork, item C — query budget)
    /// is an optional per-call wall-clock budget in seconds; when set the query
    /// is run through [`DbInstance::run_script_with_options`] and a budget expiry
    /// surfaces as an `eval::timeout` error (distinct from `::kill`'s
    /// `eval::killed`). It is appended last so existing positional/keyword
    /// callers are unaffected.
    #[pyo3(signature = (query, params, immutable, timeout=None))]
    pub fn run_script(
        &self,
        py: Python<'_>,
        query: &str,
        params: &PyDict,
        immutable: bool,
        timeout: Option<f64>,
    ) -> PyResult<PyObject> {
        // Clone the handle out of a momentary read guard and drop the guard
        // BEFORE `py.allow_threads` — holding the lock across the blocking call
        // would serialize queries and re-block `close`.
        let db = self.db_ref()?;
        let params = convert_params(params)?;
        let result = py.allow_threads(|| {
            let mutability = if immutable {
                ScriptMutability::Immutable
            } else {
                ScriptMutability::Mutable
            };
            if timeout.is_some() {
                // `..Default::default()` is intentional forward-proofing per the
                // `ScriptRunOptions` doc (new options land without touching this
                // call site); it is a no-op only while `timeout` is the sole field.
                #[allow(clippy::needless_update)]
                let opts = ScriptRunOptions {
                    timeout,
                    ..Default::default()
                };
                db.run_script_with_options(query, params, mutability, opts)
            } else {
                db.run_script(query, params, mutability)
            }
        });
        match result {
            Ok(rows) => Ok(named_rows_to_py(rows, py)),
            Err(err) => {
                let reports = format_error_as_json(err, Some(query)).to_string();
                let json_mod = py.import("json")?;
                let loads_fn = json_mod.getattr("loads")?;
                let args = PyTuple::new(py, [PyString::new(py, &reports)]);
                let msg = loads_fn.call1(args)?;
                Err(PyException::new_err(PyObject::from(msg)))
            }
        }
    }
    /// Set (or clear, with `None`) the Db-level default per-query wall-clock
    /// budget in seconds (kleos fork, item C). Forwards to
    /// [`DbInstance::set_default_query_timeout`]. The effective budget for each
    /// statement is the minimum of this default, any per-call `timeout`, and any
    /// in-script `:timeout` — so this is a guard a query cannot escape.
    pub fn set_default_query_timeout(&self, secs: Option<f64>) -> PyResult<()> {
        let db = self.db_ref()?;
        db.set_default_query_timeout(secs);
        Ok(())
    }
    /// Read the Db-level default per-query wall-clock budget in seconds, or
    /// `None` if unset (kleos fork, item C). Forwards to
    /// [`DbInstance::default_query_timeout`].
    pub fn default_query_timeout(&self) -> PyResult<Option<f64>> {
        let db = self.db_ref()?;
        Ok(db.default_query_timeout())
    }
    /// Enable or disable the automatic factorized-`count()` rewrite (kleos
    /// fork, query factorization). This is a Db-wide kill switch, **default ON
    /// since 0.13.1** — when on, a `count()` over an alpha-acyclic all-positive
    /// join is computed per-separator instead of materializing the full join,
    /// without changing the result. Forwards to
    /// [`DbInstance::set_query_factorization`]. Mirrors the Rust switch so
    /// Python callers can toggle it exactly like `set_default_query_timeout`.
    pub fn set_query_factorization(&self, enabled: bool) -> PyResult<()> {
        let db = self.db_ref()?;
        db.set_query_factorization(enabled);
        Ok(())
    }
    /// Whether the automatic factorized-`count()` rewrite is currently enabled
    /// (kleos fork, query factorization). Forwards to
    /// [`DbInstance::query_factorization`].
    pub fn query_factorization(&self) -> PyResult<bool> {
        let db = self.db_ref()?;
        Ok(db.query_factorization())
    }
    /// Request that write transactions **fsync on commit** (kleos fork,
    /// 0.14.0). Returns `True` iff the storage backend honors the knob —
    /// `rocksdb` does (per-transaction `WriteOptions.sync`; note the bulk
    /// import/restore channels are separate non-transactional paths outside
    /// it); `sqlite` returns `False` because its writes are **already
    /// durable** (`synchronous=FULL` default); `mem` has nothing to sync.
    /// The contract this makes explicit: by default the RocksDB backend
    /// writes its WAL without fsync, so an OS crash or power loss can lose
    /// the most recent acknowledged commits (a mere process crash cannot).
    /// Forwards to [`DbInstance::set_durable_writes`].
    pub fn set_durable_writes(&self, durable: bool) -> PyResult<bool> {
        let db = self.db_ref()?;
        Ok(db.set_durable_writes(durable))
    }
    /// One-call hybrid retrieval (kleos fork): HNSW + FTS (+ optional extra
    /// ranked lists and graph legs) fused with RRF and optionally diversified
    /// with MMR. Takes a dict mirroring the Rust `HybridSearch` fields;
    /// returns the same shape as `run_script` (`{rows, headers, next}`) with
    /// headers `["id","score"]` (or `["id","rank"]` when MMR is set). With
    /// `detailed: True` the output is long-format per-leg contributions:
    /// headers `["id","score","list_id","leg_rank","leg_score"]` (no MMR) or
    /// `["id","rank","score","list_id","leg_rank","leg_score"]` (MMR) — and
    /// when any graph leg sets `max_nodes` (budgeted mode), TWO further
    /// trailing columns `parent`, `depth` carry the cheapest-path witness
    /// (null on non-graph rows): 7 columns without MMR, 8 with it.
    pub fn hybrid_search(&self, py: Python<'_>, query: &PyDict) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let q = py_to_hybrid_search(query)?;
        match py.allow_threads(|| db.hybrid_search(&q)) {
            Ok(rows) => Ok(named_rows_to_py(rows, py)),
            Err(err) => {
                let reports = format_error_as_json(err, None).to_string();
                let json_mod = py.import("json")?;
                let loads_fn = json_mod.getattr("loads")?;
                let args = PyTuple::new(py, [PyString::new(py, &reports)]);
                let msg = loads_fn.call1(args)?;
                Err(PyException::new_err(PyObject::from(msg)))
            }
        }
    }
    /// Run a read-only Cypher query (kleos fork; built with the `cypher`
    /// feature). `query` is openCypher (subset); `schema` is a dict mapping the
    /// property graph onto stored relations (see `py_to_cypher_schema`); `params`
    /// supplies any `$name` parameters. Returns `{rows, headers, next}` like
    /// `run_script`, with headers = the RETURN columns.
    #[cfg(feature = "cypher")]
    pub fn run_cypher(
        &self,
        py: Python<'_>,
        query: &str,
        schema: &PyDict,
        params: &PyDict,
    ) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let sch = py_to_cypher_schema(schema)?;
        let params = convert_params(params)?;
        match py.allow_threads(|| db.run_cypher(query, &sch, params)) {
            Ok(rows) => Ok(named_rows_to_py(rows, py)),
            Err(err) => {
                let reports = format_error_as_json(err, Some(query)).to_string();
                let json_mod = py.import("json")?;
                let loads_fn = json_mod.getattr("loads")?;
                let args = PyTuple::new(py, [PyString::new(py, &reports)]);
                let msg = loads_fn.call1(args)?;
                Err(PyException::new_err(PyObject::from(msg)))
            }
        }
    }
    pub fn register_callback(&self, rel: &str, callback: &PyAny) -> PyResult<u32> {
        let db = self.db_ref()?;
        let cb: Py<PyAny> = callback.into();
        let (id, ch) = db.register_callback(rel, None);
        // A dedicated thread, not `rayon::spawn`: this loop blocks for the
        // callback's whole lifetime, and parking it on a rayon worker took that
        // worker from the global pool. With as many callbacks as cores, every
        // query that uses rayon (graph algorithms, parallel evaluation) hung.
        std::thread::spawn(move || {
            for (op, new, old) in ch {
                Python::with_gil(|py| {
                    let op = PyString::new(py, op.as_str()).into();
                    let new_py = rows_to_py_rows(new.rows, py);
                    let old_py = rows_to_py_rows(old.rows, py);
                    let args = PyTuple::new(py, [op, new_py, old_py]);
                    let callable = cb.as_ref(py);
                    if let Err(err) = callable.call1(args) {
                        eprintln!("{}", err);
                    }
                })
            }
        });
        Ok(id)
    }
    pub fn register_fixed_rule(
        &self,
        name: String,
        arity: usize,
        callback: &PyAny,
    ) -> PyResult<()> {
        let db = self.db_ref()?;
        let cb: Py<PyAny> = callback.into();
        let rule_impl = SimpleFixedRule::new(arity, move |inputs, options| -> Result<_> {
            Python::with_gil(|py| -> Result<NamedRows> {
                let py_inputs = PyList::new(
                    py,
                    inputs.into_iter().map(|nr| rows_to_py_rows(nr.rows, py)),
                );
                let py_opts = options_to_py(options, py).into_diagnostic()?;
                let args = PyTuple::new(py, vec![PyObject::from(py_inputs), py_opts]);
                let res = cb.as_ref(py).call1(args).into_diagnostic()?;
                Ok(NamedRows::new(vec![], py_to_rows(res).into_diagnostic()?))
            })
        });
        db.register_fixed_rule(name, rule_impl).map_err(report2py)
    }
    /// Register a re-rank model (kleos fork): `callback(query: str,
    /// docs: list[str]) -> list[float]`, one score per doc, higher = more
    /// relevant. Usable as `Rerank(..., model: name, ...)` in CozoScript and
    /// as `hybrid_search({..., "rerank": {"model": name, "top_n": 50}})`.
    pub fn register_reranker(&self, name: String, callback: &PyAny) -> PyResult<()> {
        let db = self.db_ref()?;
        db.register_reranker(name, std::sync::Arc::new(PyReranker(callback.into())))
            .map_err(report2py)
    }
    /// Unregister a re-rank model. Returns whether one was registered.
    pub fn unregister_reranker(&self, name: &str) -> PyResult<bool> {
        let db = self.db_ref()?;
        Ok(db.unregister_reranker(name))
    }
    /// Live query subscription (kleos, UPGRADE_PLAN F1;
    /// `docs/specs/live-subscriptions.md`). `callback(event: dict)` runs on a
    /// background thread for every event: `{"kind": "initial" | "resync",
    /// "headers", "rows"}`, `{"kind": "delta", "headers", "added", "removed"}`
    /// or `{"kind": "error", "message"}`. Returns an id for `unsubscribe`.
    #[pyo3(signature = (query, params, callback, queue=None))]
    pub fn subscribe(
        &self,
        py: Python<'_>,
        query: &str,
        params: &PyDict,
        callback: &PyAny,
        queue: Option<usize>,
    ) -> PyResult<u64> {
        let db = self.db_ref()?;
        let params = convert_params(params)?;
        let opts = SubscribeOptions {
            queue_capacity: queue.unwrap_or(SubscribeOptions::default().queue_capacity),
        };
        let sub = py
            .allow_threads(|| db.subscribe(query, params, opts))
            .map_err(report2py)?;
        let id = sub.id();
        self.subs.lock().unwrap().insert(id);
        let cb: Py<PyAny> = callback.into();
        std::thread::spawn(move || {
            while let Some(ev) = sub.recv() {
                Python::with_gil(|py| {
                    let rows_ev = |kind: &str, rows: &NamedRows| -> PyObject {
                        BTreeMap::from([
                            ("kind", kind.into_py(py)),
                            ("headers", rows.headers.clone().into_py(py)),
                            ("rows", rows_to_py_rows(rows.rows.clone(), py)),
                        ])
                        .into_py(py)
                    };
                    let event: PyObject = match ev {
                        SubscriptionEvent::Initial { rows } => rows_ev("initial", &rows),
                        SubscriptionEvent::Resync { rows } => rows_ev("resync", &rows),
                        SubscriptionEvent::Delta { added, removed } => BTreeMap::from([
                            ("kind", "delta".into_py(py)),
                            ("headers", added.headers.clone().into_py(py)),
                            ("added", rows_to_py_rows(added.rows.clone(), py)),
                            ("removed", rows_to_py_rows(removed.rows.clone(), py)),
                        ])
                        .into_py(py),
                        SubscriptionEvent::Error { message } => BTreeMap::from([
                            ("kind", "error".into_py(py)),
                            ("message", message.into_py(py)),
                        ])
                        .into_py(py),
                    };
                    if let Err(err) = cb.as_ref(py).call1((event,)) {
                        eprintln!("{}", err);
                    }
                })
            }
        });
        Ok(id)
    }
    /// End a subscription made with `subscribe`. Returns whether it existed.
    pub fn unsubscribe(&self, id: u64) -> bool {
        self.subs.lock().unwrap().remove(&id);
        let maybe_db = self.db.read().unwrap().clone();
        match maybe_db {
            Some(db) => db.unsubscribe(id),
            None => false,
        }
    }
    pub fn unregister_callback(&self, id: u32) -> bool {
        // Preserve the pre-fork behavior: a closed handle returns `false`,
        // not an error. Clone out of a momentary guard, drop it, then call.
        let maybe_db = self.db.read().unwrap().clone();
        match maybe_db {
            Some(db) => db.unregister_callback(id),
            None => false,
        }
    }
    pub fn unregister_fixed_rule(&self, name: &str) -> PyResult<bool> {
        // Preserve the pre-fork behavior: a closed handle returns `Ok(false)`,
        // not an error.
        let maybe_db = self.db.read().unwrap().clone();
        match maybe_db {
            Some(db) => match db.unregister_fixed_rule(name) {
                Ok(b) => Ok(b),
                Err(err) => Err(PyException::new_err(err.to_string())),
            },
            None => Ok(false),
        }
    }
    /// Set the ceiling, in bytes, on the total size of cached graph projections
    /// (kleos fork; default 512 MiB).
    ///
    /// Enforced immediately: variants are evicted least-recently-used first
    /// until the cache fits. `0` evicts everything and turns caching off, while
    /// `::graph create`/`list`/`drop` keep working and every algorithm builds
    /// its adjacency fresh. Without this the ceiling was unreachable from
    /// Python, yet the engine's oversize-variant warning names it as the
    /// remedy.
    #[cfg(feature = "graph-algo")]
    pub fn set_graph_projection_capacity(&self, bytes: usize) -> PyResult<()> {
        let db = self.db_ref()?;
        db.set_graph_projection_capacity(bytes);
        Ok(())
    }
    pub fn export_relations(&self, py: Python<'_>, relations: Vec<String>) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let res = match py.allow_threads(|| db.export_relations(relations.iter())) {
            Ok(res) => res,
            Err(err) => return Err(PyException::new_err(err.to_string())),
        };
        let ret = PyDict::new(py);
        for (k, v) in res {
            ret.set_item(k, named_rows_to_py(v, py))?;
        }
        Ok(ret.into())
    }
    pub fn import_relations(&self, py: Python<'_>, data: &PyDict) -> PyResult<()> {
        let db = self.db_ref()?;
        let mut arg = BTreeMap::new();
        for (k, v) in data.iter() {
            let k = k.extract::<String>()?;
            let vals = py_to_named_rows(v)?;
            arg.insert(k, vals);
        }
        py.allow_threads(|| db.import_relations(arg))
            .map_err(report2py)
    }
    /// Atomically copy a local Parquet or Arrow IPC file into an existing
    /// stored relation (kleos fork; feature `columnar-io`).
    #[cfg(feature = "columnar-io")]
    #[pyo3(signature = (
        relation,
        path,
        *,
        format,
        columns=None,
        batch_rows=8192,
        timeout=None,
        max_source_bytes=None,
        max_rows=None,
        max_decoded_batch_bytes=None,
        max_value_bytes=None,
        max_nesting_depth=16
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn import_columnar_file(
        &self,
        py: Python<'_>,
        relation: &str,
        path: &Bound<'_, PyAny>,
        format: &str,
        columns: Option<BTreeMap<String, String>>,
        batch_rows: usize,
        timeout: Option<f64>,
        max_source_bytes: Option<u64>,
        max_rows: Option<u64>,
        max_decoded_batch_bytes: Option<usize>,
        max_value_bytes: Option<usize>,
        max_nesting_depth: usize,
    ) -> PyResult<PyObject> {
        let format = match format {
            "parquet" => ColumnarFileFormat::Parquet,
            "arrow_ipc_file" => ColumnarFileFormat::ArrowIpcFile,
            "arrow_ipc_stream" => ColumnarFileFormat::ArrowIpcStream,
            other => {
                return Err(PyException::new_err(format!(
                    "unknown columnar format '{other}'; expected parquet, arrow_ipc_file, or arrow_ipc_stream"
                )))
            }
        };
        let timeout = match timeout {
            Some(seconds) if seconds.is_finite() && seconds > 0.0 => {
                Some(Duration::from_secs_f64(seconds))
            }
            Some(_) => {
                return Err(PyException::new_err(
                    "timeout must be a finite number greater than zero",
                ))
            }
            None => None,
        };
        let path: String = py
            .import_bound("os")?
            .getattr("fspath")?
            .call1((path,))?
            .extract()?;
        let options = ColumnarImportOptions::new(format)
            .with_columns(columns.unwrap_or_default())
            .with_batch_rows(batch_rows)
            .with_timeout(timeout)
            .with_max_source_bytes(max_source_bytes)
            .with_max_rows(max_rows)
            .with_max_decoded_batch_bytes(max_decoded_batch_bytes)
            .with_max_value_bytes(max_value_bytes)
            .with_max_nesting_depth(max_nesting_depth);
        let db = self.db_ref()?;
        let report = py
            .allow_threads(|| db.import_columnar_file(relation, path, &options))
            .map_err(report2py)?;
        let result = PyDict::new_bound(py);
        result.set_item("rows_processed", report.rows_processed)?;
        result.set_item("batches_processed", report.batches_processed)?;
        result.set_item(
            "search_indexes_requiring_rebuild",
            report.search_indexes_requiring_rebuild,
        )?;
        Ok(result.into_py(py))
    }
    pub fn backup(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let db = self.db_ref()?;
        py.allow_threads(|| db.backup_db(path)).map_err(report2py)
    }
    pub fn restore(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let db = self.db_ref()?;
        py.allow_threads(|| db.restore_backup(path))
            .map_err(report2py)
    }
    pub fn import_from_backup(
        &self,
        py: Python<'_>,
        in_file: &str,
        relations: Vec<String>,
    ) -> PyResult<()> {
        let db = self.db_ref()?;
        py.allow_threads(|| db.import_from_backup(in_file, &relations))
            .map_err(report2py)
    }
    /// Close the handle (kleos fork, item D). Takes `&self` (via a write guard)
    /// so it no longer collides with an in-flight `run_script(&self)`'s shared
    /// PyCell borrow — the "Already borrowed" bug. Returns immediately even while
    /// queries run; the underlying storage drops when the last in-flight
    /// `DbInstance` clone is dropped.
    ///
    /// Live subscriptions made through this handle are ended first, so their
    /// delivery threads exit and release their callbacks.
    pub fn close(&self) -> bool {
        let Some(db) = self.db.write().unwrap().take() else {
            return false;
        };
        let subs = std::mem::take(&mut *self.subs.lock().unwrap());
        for id in subs {
            db.unsubscribe(id);
        }
        true
    }
    /// Turn a built-in embedder on under `name` (kleos fork; process-wide,
    /// like `::embedder enable`). With no keyword arguments a previously
    /// disabled embedder comes back as it was; otherwise `model` defaults to
    /// EmbeddingGemma (`'embeddinggemma'`, int8, auto device, 768-d), or e.g.
    /// `'potion-base-8M'`. Built-in models need a build with the `infer`
    /// feature.
    #[pyo3(signature = (name, model=None, precision=None, device=None, dim=None))]
    pub fn enable_embedder(
        &self,
        py: Python<'_>,
        name: &str,
        model: Option<String>,
        precision: Option<String>,
        device: Option<String>,
        dim: Option<usize>,
    ) -> PyResult<()> {
        self.db_ref()?;
        let spec = ::kleos::embed::EmbedderSpec {
            model,
            precision,
            device,
            dim,
        };
        py.allow_threads(|| ::kleos::embed::enable_embedder(name, spec))
            .map_err(report2py)
    }
    /// Turn the embedder `name` off: unregister it (and gemma's `_query`
    /// twin), unload its model and drop its cached vectors. Returns `False`
    /// when it was already off. Process-wide.
    pub fn disable_embedder(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        self.db_ref()?;
        py.allow_threads(|| ::kleos::embed::disable_embedder(name))
            .map_err(report2py)
    }
    /// Every enabled and disabled embedder as a dict with `name`, `kind`,
    /// `model`, `dim`, `enabled`, `loaded`, `precision`, `device` (the module
    /// function `registered_embedders()` keeps returning `(name, dim)` of the
    /// enabled ones).
    pub fn embedders(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.db_ref()?;
        let out = PyList::empty(py);
        for e in ::kleos::embed::embedders() {
            let d = PyDict::new(py);
            d.set_item("name", e.name)?;
            d.set_item("kind", e.kind)?;
            d.set_item("model", e.model)?;
            d.set_item("dim", e.dim)?;
            d.set_item("enabled", e.enabled)?;
            d.set_item("loaded", e.loaded)?;
            d.set_item("precision", e.precision)?;
            d.set_item("device", e.device)?;
            out.append(d)?;
        }
        Ok(out.into())
    }
    /// Fill the null embedded columns of `relation` (columns whose default
    /// calls `embed()` / `embed_or_null()`), batched through the cache; same
    /// as `::embedder backfill <relation>`. Returns the usual result dict with
    /// rows `[column, filled, still_null]`.
    pub fn backfill_embeddings(&self, py: Python<'_>, relation: &str) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let rows = py
            .allow_threads(|| db.backfill_embeddings(relation))
            .map_err(report2py)?;
        Ok(named_rows_to_py(rows, py))
    }
    /// Turn a built-in decision model on under `name` (kleos fork;
    /// process-wide, like `::decider enable`). `model` is `'decider-nano'`
    /// (default; kleos's fine-tune `Mazino0/kleos-decider-nano`),
    /// `'decider-nano-base'` (upstream `tasksource/tasksource-decider-nano`)
    /// or `'d1-omni'`; needs a build with the `infer` feature. With
    /// no keyword arguments a disabled decider comes back as it was.
    /// `path` loads decider-nano weights from a local directory (e.g. a
    /// fine-tune) instead of the Hugging Face cache.
    #[pyo3(signature = (name, model=None, precision=None, device=None, path=None))]
    pub fn enable_decider(
        &self,
        py: Python<'_>,
        name: &str,
        model: Option<String>,
        precision: Option<String>,
        device: Option<String>,
        path: Option<String>,
    ) -> PyResult<()> {
        self.db_ref()?;
        let spec = ::kleos::decide::DeciderSpec {
            model,
            path,
            precision,
            device,
        };
        py.allow_threads(|| ::kleos::decide::enable_decider(name, spec))
            .map_err(report2py)
    }
    /// Turn the decider `name` off and unload it. Returns `False` when it was
    /// already off. Process-wide.
    pub fn disable_decider(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        self.db_ref()?;
        py.allow_threads(|| ::kleos::decide::disable_decider(name))
            .map_err(report2py)
    }
    /// Every enabled and disabled decider as a dict.
    pub fn deciders(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.db_ref()?;
        let out = PyList::empty(py);
        for e in ::kleos::decide::deciders() {
            let d = PyDict::new(py);
            d.set_item("name", e.name)?;
            d.set_item("kind", e.kind)?;
            d.set_item("family", e.family)?;
            d.set_item("model", e.model)?;
            d.set_item("enabled", e.enabled)?;
            d.set_item("loaded", e.loaded)?;
            d.set_item("precision", e.precision)?;
            d.set_item("device", e.device)?;
            out.append(d)?;
        }
        Ok(out.into())
    }
    /// Decide with a fixed template (`supersedes`, `same_entity`,
    /// `duplicate`, `contradicts`, `relation`, `important`). Returns a dict
    /// `{label, p, probs, band, confident, model, template, template_version,
    /// via, reason, decided_by}`. `b` is omitted for `important`.
    #[pyo3(signature = (template, a, b=None, model="nano"))]
    pub fn decide(
        &self,
        py: Python<'_>,
        template: &str,
        a: &str,
        b: Option<&str>,
        model: &str,
    ) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let b = b.unwrap_or("");
        let d = py
            .allow_threads(|| db.decide(model, template, a, b))
            .map_err(report2py)?;
        let out = PyDict::new(py);
        out.set_item("decided_by", d.decided_by())?;
        out.set_item("label", d.label)?;
        out.set_item("p", d.p)?;
        out.set_item("probs", d.probs)?;
        out.set_item("band", d.band.name())?;
        out.set_item("confident", d.confident)?;
        out.set_item("model", d.model)?;
        out.set_item("template", d.template)?;
        out.set_item("template_version", d.template_version)?;
        out.set_item("via", d.via)?;
        out.set_item("reason", d.reason)?;
        Ok(out.into())
    }
    /// Resolve entity mentions against the entity relation `relation`
    /// (docs/specs/entity-resolution.md). `mentions` is a list of names or
    /// dicts with `name`, `type`, `aliases`, `description`, `context`, `id`.
    /// Options are the `::entity resolve` keys (`type`, `aliases`,
    /// `description`, `embedding`, `lsh`, `fts`, `hnsw`, `embedder`,
    /// `decider`, `k`, `max_candidates`, `threshold`, `tau_auto`, ...);
    /// `commit=True` writes rows, merges and review links. Returns one dict
    /// per mention: `{mention, name, outcome, entity, id, p, band,
    /// decided_by, reason, candidates}`.
    #[pyo3(signature = (relation, mentions, commit=false, **options))]
    pub fn resolve_entities(
        &self,
        py: Python<'_>,
        relation: &str,
        mentions: &PyList,
        commit: bool,
        options: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let mut cfg = EntityConfig::new(relation);
        if let Some(o) = options {
            for (k, v) in o.iter() {
                let k: String = k.extract()?;
                let v = py_to_value(v)?;
                cfg.set(&k, &v).map_err(report2py)?;
            }
        }
        let mut ms = Vec::with_capacity(mentions.len());
        for m in mentions.iter() {
            let mention = if let Ok(name) = m.extract::<String>() {
                Mention::new(name)
            } else {
                let d: &PyDict = m.downcast()?;
                let get_str = |k: &str| -> PyResult<Option<String>> {
                    Ok(match d.get_item(k)? {
                        Some(x) if !x.is_none() => Some(x.extract()?),
                        _ => None,
                    })
                };
                let name = get_str("name")?
                    .ok_or_else(|| PyException::new_err("a mention dict needs 'name'"))?;
                let mut x = Mention::new(name);
                x.entity_type = get_str("type")?;
                x.description = get_str("description")?;
                x.context = get_str("context")?;
                x.id = match d.get_item("id")? {
                    Some(v) if !v.is_none() => Some(py_to_value(v)?),
                    _ => None,
                };
                if let Some(a) = d.get_item("aliases")? {
                    x.aliases = a.extract()?;
                }
                x
            };
            ms.push(mention);
        }
        let res = py
            .allow_threads(|| db.resolve_entities(&cfg, &ms, ResolveOptions { commit }))
            .map_err(report2py)?;
        let out = PyList::empty(py);
        for r in res {
            let d = PyDict::new(py);
            d.set_item("mention", r.mention)?;
            d.set_item("name", r.name)?;
            d.set_item("outcome", r.outcome.name())?;
            d.set_item("entity", r.entity.map(|e| value_to_py(e, py)))?;
            d.set_item("id", r.id.map(|e| value_to_py(e, py)))?;
            d.set_item("p", r.p)?;
            d.set_item("band", r.band.name())?;
            d.set_item("decided_by", r.decided_by)?;
            d.set_item("reason", r.reason)?;
            let cands = PyList::empty(py);
            for c in r.candidates {
                let cd = PyDict::new(py);
                cd.set_item("entity", value_to_py(c.entity, py))?;
                cd.set_item("name", c.name)?;
                cd.set_item("sources", c.sources)?;
                cd.set_item("rule", c.rule)?;
                cd.set_item("jaccard", c.jaccard)?;
                cd.set_item("p", c.p)?;
                cd.set_item("band", c.band.map(|b| b.name()))?;
                cd.set_item("decided_by", c.decided_by)?;
                cands.append(cd)?;
            }
            d.set_item("candidates", cands)?;
            out.append(d)?;
        }
        Ok(out.into())
    }
    /// Merge entity `source` into `target` by hand (accepting a review link).
    #[pyo3(signature = (relation, source, target, reason="manual merge"))]
    pub fn merge_entities(&self, py: Python<'_>, relation: &str, source: &PyAny, target: &PyAny, reason: &str) -> PyResult<()> {
        let db = self.db_ref()?;
        let (a, b) = (py_to_value(source)?, py_to_value(target)?);
        py.allow_threads(|| db.merge_entities(relation, &a, &b, reason)).map_err(report2py)
    }
    /// Undo the merge of `id`; returns what it had been merged into (or None).
    pub fn split_entity(&self, py: Python<'_>, relation: &str, id: &PyAny) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let id = py_to_value(id)?;
        let r = py.allow_threads(|| db.split_entity(relation, &id)).map_err(report2py)?;
        Ok(match r {
            Some(v) => value_to_py(v, py),
            None => py.None(),
        })
    }
    /// The canonical entity of `id`.
    pub fn canonical_entity(&self, py: Python<'_>, relation: &str, id: &PyAny) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let id = py_to_value(id)?;
        let r = py.allow_threads(|| db.canonical_entity(relation, &id)).map_err(report2py)?;
        Ok(value_to_py(r, py))
    }
    /// Merge history (of `id` when given): list of dicts `{from, into,
    /// action, at, p, decided_by, reason}`, oldest first.
    #[pyo3(signature = (relation, id=None))]
    pub fn entity_history(&self, py: Python<'_>, relation: &str, id: Option<&PyAny>) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let id = match id {
            Some(v) => Some(py_to_value(v)?),
            None => None,
        };
        let ev = py.allow_threads(|| db.entity_history(relation, id.as_ref())).map_err(report2py)?;
        let out = PyList::empty(py);
        for e in ev {
            let d = PyDict::new(py);
            d.set_item("from", value_to_py(e.from, py))?;
            d.set_item("into", value_to_py(e.into, py))?;
            d.set_item("action", e.action)?;
            d.set_item("at", e.at)?;
            d.set_item("p", e.p)?;
            d.set_item("decided_by", e.decided_by)?;
            d.set_item("reason", e.reason)?;
            out.append(d)?;
        }
        Ok(out.into())
    }
    /// Possible same-as links awaiting review (named rows).
    pub fn entity_pending(&self, py: Python<'_>, relation: &str) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let rows = py.allow_threads(|| db.entity_pending(relation)).map_err(report2py)?;
        Ok(named_rows_to_py(rows, py))
    }
    /// Fit a logistic probe (or with `probe=False` a calibration) for decider
    /// `name` and `template` on the labelled stored relation `relation` (last
    /// column = label); same as `::decider fit`. Persisted in this database.
    #[pyo3(signature = (name, template, relation, probe=true, alpha=0.02))]
    pub fn fit_decider(
        &self,
        py: Python<'_>,
        name: &str,
        template: &str,
        relation: &str,
        probe: bool,
        alpha: f64,
    ) -> PyResult<PyObject> {
        let db = self.db_ref()?;
        let r = py
            .allow_threads(|| db.fit_decider(name, template, relation, probe, alpha))
            .map_err(report2py)?;
        let out = PyDict::new(py);
        out.set_item("model", r.model)?;
        out.set_item("template", r.template)?;
        out.set_item("kind", r.kind)?;
        out.set_item("n", r.n)?;
        out.set_item("cv_score", r.cv_score)?;
        out.set_item("threshold", r.threshold)?;
        out.set_item("tau_auto", r.tau_auto)?;
        out.set_item("coverage", r.coverage)?;
        Ok(out.into())
    }
    /// Create the agent memory schema (kleos fork; docs/specs/memory-api.md)
    /// or update it: `embedder=` attaches an embedder and backfills vectors,
    /// `decider=` sets the default decider. Returns the configuration.
    #[pyo3(signature = (**opts))]
    pub fn memory_init(&self, py: Python<'_>, opts: Option<&PyDict>) -> PyResult<PyObject> {
        self.memory_op(py, "init", vec![], opts)
    }
    /// Store a memory. Options: `ns`, `kind`, `source`, `importance`,
    /// `entities` (names), `at` (valid-from µs), `k`, `decider`, `embedder`.
    /// Returns `{action, id, superseded, duplicate_of, tentative, decisions,
    /// entities, embedded, decider, ns}`; decisions carry `decided_by`.
    #[pyo3(signature = (text, **opts))]
    pub fn remember(&self, py: Python<'_>, text: &str, opts: Option<&PyDict>) -> PyResult<PyObject> {
        self.memory_op(py, "remember", vec![("text", json!(text))], opts)
    }
    /// Hybrid recall over current memories (or those valid at `as_of`).
    /// Options: `ns`, `k`, `kind`, `entity`, `as_of`, `vector_weight`.
    /// Returns a list of `{id, text, score, kind, valid_from, valid_to, ...}`.
    #[pyo3(signature = (query, **opts))]
    pub fn recall(&self, py: Python<'_>, query: &str, opts: Option<&PyDict>) -> PyResult<PyObject> {
        self.memory_op(py, "recall", vec![("query", json!(query))], opts)
    }
    /// Forget a memory (`id`) or a filter (`ids`, `kind`, `entity`,
    /// `all=True`). `mode='retract'` (default) ends validity; `mode='erase'`
    /// deletes every version (GDPR).
    #[pyo3(signature = (id=None, **opts))]
    pub fn forget(&self, py: Python<'_>, id: Option<&str>, opts: Option<&PyDict>) -> PyResult<PyObject> {
        let extra = id.map(|i| vec![("id", json!(i))]).unwrap_or_default();
        self.memory_op(py, "forget", extra, opts)
    }
    /// Pending tentative decisions, recent changes and contradiction
    /// candidates of a namespace (model-free). Options: `since`, `limit`.
    #[pyo3(signature = (ns=None, **opts))]
    pub fn reflect(&self, py: Python<'_>, ns: Option<&str>, opts: Option<&PyDict>) -> PyResult<PyObject> {
        let extra = ns.map(|n| vec![("ns", json!(n))]).unwrap_or_default();
        self.memory_op(py, "reflect", extra, opts)
    }
    /// Typed link `a -[rel]-> b` between memories or entities. Options:
    /// `ns`, `p`.
    #[pyo3(signature = (a, b, rel, **opts))]
    pub fn link(&self, py: Python<'_>, a: &str, b: &str, rel: &str, opts: Option<&PyDict>) -> PyResult<PyObject> {
        self.memory_op(
            py,
            "link",
            vec![("from", json!(a)), ("to", json!(b)), ("rel", json!(rel))],
            opts,
        )
    }
    pub fn multi_transact(&self, write: bool) -> PyResult<CozoDbMulTx> {
        let db = self.db_ref()?;
        Ok(CozoDbMulTx {
            tx: db.multi_transaction(write),
        })
    }
}

#[pymethods]
impl CozoDbMulTx {
    pub fn abort(&self) -> PyResult<()> {
        self.tx
            .abort()
            .map_err(|err| PyException::new_err(err.to_string()))
    }
    pub fn commit(&self) -> PyResult<()> {
        self.tx
            .commit()
            .map_err(|err| PyException::new_err(err.to_string()))
    }
    pub fn run_script(&self, py: Python<'_>, query: &str, params: &PyDict) -> PyResult<PyObject> {
        let params = convert_params(params)?;
        match py.allow_threads(|| self.tx.run_script(query, params)) {
            Ok(rows) => Ok(named_rows_to_py(rows, py)),
            Err(err) => {
                let reports = format_error_as_json(err, Some(query)).to_string();
                let json_mod = py.import("json")?;
                let loads_fn = json_mod.getattr("loads")?;
                let args = PyTuple::new(py, [PyString::new(py, &reports)]);
                let msg = loads_fn.call1(args)?;
                Err(PyException::new_err(PyObject::from(msg)))
            }
        }
    }
}

#[pyfunction]
fn eval_expressions(
    py: Python<'_>,
    query: &str,
    params: &PyDict,
    bindings: &PyDict,
) -> PyResult<PyObject> {
    let params = convert_params(params).unwrap();
    let bindings = convert_params(bindings).unwrap();
    match evaluate_expressions(query, &params, &bindings) {
        Ok(v) => Ok(value_to_py(v, py)),
        Err(err) => {
            let reports = format_error_as_json(err, Some(query)).to_string();
            let json_mod = py.import("json")?;
            let loads_fn = json_mod.getattr("loads")?;
            let args = PyTuple::new(py, [PyString::new(py, &reports)]);
            let msg = loads_fn.call1(args)?;
            Err(PyException::new_err(PyObject::from(msg)))
        }
    }
}

#[pyfunction]
fn variables(py: Python<'_>, query: &str, params: &PyDict) -> PyResult<BTreeSet<String>> {
    let params = convert_params(params).unwrap();
    match get_variables(query, &params) {
        Ok(rows) => Ok(rows),
        Err(err) => {
            let reports = format_error_as_json(err, Some(query)).to_string();
            let json_mod = py.import("json")?;
            let loads_fn = json_mod.getattr("loads")?;
            let args = PyTuple::new(py, [PyString::new(py, &reports)]);
            let msg = loads_fn.call1(args)?;
            Err(PyException::new_err(PyObject::from(msg)))
        }
    }
}

/// Register a Python embedder (kleos fork): `callback(texts: list[str]) ->
/// list[list[float]]`, one `dim`-wide vector per text, in order. Callable from
/// CozoScript as `embed(name, text)` and usable in computed columns
/// (`vec: <F32; dim> default embed(name, text)`). The registry is
/// process-wide (shared by every open database). `max_batch` caps the texts per
/// call; `model_id`, if given, is the embedding-cache identity (vectors cached
/// under it are reused across re-registrations), otherwise each registration
/// gets a fresh cache identity. Replaces an embedder already registered under
/// `name`.
#[pyfunction]
#[pyo3(signature = (name, callback, dim, max_batch=32, model_id=None))]
#[pyo3(name = "register_embedder")]
fn py_register_embedder(
    name: &str,
    callback: &PyAny,
    dim: usize,
    max_batch: usize,
    model_id: Option<String>,
) -> PyResult<()> {
    if !callback.is_callable() {
        return Err(PyException::new_err("register_embedder: callback is not callable"));
    }
    let e = PyEmbedder {
        callback: callback.into(),
        dim,
        max_batch: max_batch.max(1),
        model_id,
    };
    ::kleos::embed::register_embedder(name, std::sync::Arc::new(e)).map_err(report2py)
}

/// Remove an embedder registered under `name`; returns whether one existed.
#[pyfunction]
#[pyo3(name = "unregister_embedder")]
fn py_unregister_embedder(name: &str) -> bool {
    ::kleos::embed::unregister_embedder(name)
}

/// `[(name, dim)]` of every registered embedder (Python and built-in).
#[pyfunction]
#[pyo3(name = "registered_embedders")]
fn py_registered_embedders() -> Vec<(String, usize)> {
    ::kleos::embed::registered_embedders()
}

/// Module-level twin of `CozoDbPy.enable_embedder` (the registry is
/// process-wide, so no handle is needed).
#[pyfunction]
#[pyo3(name = "enable_embedder", signature = (name, model=None, precision=None, device=None, dim=None))]
fn py_enable_embedder(
    py: Python<'_>,
    name: &str,
    model: Option<String>,
    precision: Option<String>,
    device: Option<String>,
    dim: Option<usize>,
) -> PyResult<()> {
    let spec = ::kleos::embed::EmbedderSpec {
        model,
        precision,
        device,
        dim,
    };
    py.allow_threads(|| ::kleos::embed::enable_embedder(name, spec))
        .map_err(report2py)
}

/// Module-level twin of `CozoDbPy.disable_embedder`.
#[pyfunction]
#[pyo3(name = "disable_embedder")]
fn py_disable_embedder(py: Python<'_>, name: &str) -> PyResult<bool> {
    py.allow_threads(|| ::kleos::embed::disable_embedder(name))
        .map_err(report2py)
}

#[pymodule]
fn kleos(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<CozoDbPy>()?;
    m.add_class::<CozoDbMulTx>()?;
    m.add_function(wrap_pyfunction!(eval_expressions, m)?)?;
    m.add_function(wrap_pyfunction!(variables, m)?)?;
    m.add_function(wrap_pyfunction!(py_register_embedder, m)?)?;
    m.add_function(wrap_pyfunction!(py_unregister_embedder, m)?)?;
    m.add_function(wrap_pyfunction!(py_registered_embedders, m)?)?;
    m.add_function(wrap_pyfunction!(py_enable_embedder, m)?)?;
    m.add_function(wrap_pyfunction!(py_disable_embedder, m)?)?;
    Ok(())
}
