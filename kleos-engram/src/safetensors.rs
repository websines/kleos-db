/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The overlay export format: one small safetensors file per Engram layer.
//!
//! ```text
//! row_ids  I64  [n]          sorted ascending, unique
//! values   BF16 [n, row_dim]  replacement rows (the engine reads these instead of base rows)
//! __metadata__: format = "kleos-engram-overlay/v1", fingerprint, layer, row_dim, ...
//! ```
//!
//! Written and read here without a safetensors dependency (the format is an
//! 8-byte little-endian header length, a JSON header, then raw bytes).

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::{EngramError, Result};

/// Format tag stored in the metadata.
pub const OVERLAY_FORMAT: &str = "kleos-engram-overlay/v1";

/// One layer of a materialised overlay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlayLayer {
    /// Transformer layer id.
    pub layer: u64,
    /// Row dimension.
    pub row_dim: usize,
    /// Row ids, ascending.
    pub rows: Vec<i64>,
    /// bf16 bits, `rows.len() * row_dim`.
    pub values: Vec<u16>,
}

fn e<T: std::fmt::Display>(what: &str) -> impl FnOnce(T) -> EngramError + '_ {
    move |x| EngramError(format!("{what}: {x}"))
}

/// Serialise one layer (metadata values must be strings in safetensors).
pub fn to_bytes(layer: &OverlayLayer, metadata: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    let n = layer.rows.len();
    if layer.values.len() != n * layer.row_dim {
        return Err(EngramError(format!(
            "overlay layer {}: {} values for {n} rows of dim {}",
            layer.layer,
            layer.values.len(),
            layer.row_dim
        )));
    }
    let ids_len = n * 8;
    let vals_len = n * layer.row_dim * 2;
    let mut meta = Map::new();
    for (k, v) in metadata {
        meta.insert(k.clone(), Value::String(v.clone()));
    }
    meta.insert("format".into(), Value::String(OVERLAY_FORMAT.into()));
    meta.insert("layer".into(), Value::String(layer.layer.to_string()));
    meta.insert("row_dim".into(), Value::String(layer.row_dim.to_string()));
    let header = json!({
        "__metadata__": meta,
        "row_ids": {"dtype": "I64", "shape": [n], "data_offsets": [0, ids_len]},
        "values": {"dtype": "BF16", "shape": [n, layer.row_dim], "data_offsets": [ids_len, ids_len + vals_len]},
    });
    let mut h = serde_json::to_vec(&header).map_err(e("encoding the header"))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + ids_len + vals_len);
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    for r in &layer.rows {
        out.extend_from_slice(&r.to_le_bytes());
    }
    for v in &layer.values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

/// Write one layer to `path`.
pub fn write_file(path: &Path, layer: &OverlayLayer, metadata: &BTreeMap<String, String>) -> Result<()> {
    let bytes = to_bytes(layer, metadata)?;
    let tmp = path.with_extension("safetensors.tmp");
    let mut f = std::fs::File::create(&tmp).map_err(e("creating the export file"))?;
    f.write_all(&bytes).map_err(e("writing the export file"))?;
    f.sync_all().map_err(e("syncing the export file"))?;
    std::fs::rename(&tmp, path).map_err(e("renaming the export file"))?;
    Ok(())
}

/// Parse a layer file written by [`to_bytes`].
pub fn from_bytes(b: &[u8]) -> Result<(OverlayLayer, BTreeMap<String, String>)> {
    let bad = |m: &str| EngramError(format!("not a kleos Engram overlay file: {m}"));
    if b.len() < 8 {
        return Err(bad("too short"));
    }
    let hl = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    if b.len() < 8 + hl {
        return Err(bad("truncated header"));
    }
    let header: Value = serde_json::from_slice(&b[8..8 + hl]).map_err(e("parsing the header"))?;
    let data = &b[8 + hl..];
    let meta: BTreeMap<String, String> = header
        .get("__metadata__")
        .and_then(Value::as_object)
        .map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect())
        .unwrap_or_default();
    if meta.get("format").map(String::as_str) != Some(OVERLAY_FORMAT) {
        return Err(bad("missing format tag"));
    }
    let span = |name: &str, dtype: &str| -> Result<(&[u8], Vec<usize>)> {
        let t = header.get(name).ok_or_else(|| bad(&format!("no tensor '{name}'")))?;
        if t.get("dtype").and_then(Value::as_str) != Some(dtype) {
            return Err(bad(&format!("tensor '{name}' is not {dtype}")));
        }
        let off: Vec<usize> = t["data_offsets"].as_array().ok_or_else(|| bad("offsets"))?.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        let shape: Vec<usize> = t["shape"].as_array().ok_or_else(|| bad("shape"))?.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        if off.len() != 2 || off[1] < off[0] || off[1] > data.len() {
            return Err(bad("offsets out of range"));
        }
        Ok((&data[off[0]..off[1]], shape))
    };
    let (ids, ishape) = span("row_ids", "I64")?;
    let (vals, vshape) = span("values", "BF16")?;
    let n = *ishape.first().unwrap_or(&0);
    let row_dim = *vshape.get(1).unwrap_or(&0);
    if ids.len() != n * 8 || vals.len() != n * row_dim * 2 || vshape.first() != Some(&n) {
        return Err(bad("tensor sizes disagree"));
    }
    let layer = meta.get("layer").and_then(|s| s.parse().ok()).ok_or_else(|| bad("no layer"))?;
    Ok((
        OverlayLayer {
            layer,
            row_dim,
            rows: ids.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            values: vals.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        },
        meta,
    ))
}

/// Read one layer file.
pub fn read_file(path: &Path) -> Result<(OverlayLayer, BTreeMap<String, String>)> {
    let mut b = vec![];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut b))
        .map_err(e("reading the overlay file"))?;
    from_bytes(&b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let l = OverlayLayer { layer: 3, row_dim: 2, rows: vec![5, 9], values: vec![1, 2, 3, 4] };
        let meta = BTreeMap::from([("fingerprint".to_string(), "eg1:00".to_string())]);
        let b = to_bytes(&l, &meta).unwrap();
        let hl = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        assert_eq!(hl % 8, 0);
        let (back, m) = from_bytes(&b).unwrap();
        assert_eq!(back, l);
        assert_eq!(m["fingerprint"], "eg1:00");
        assert_eq!(m["format"], OVERLAY_FORMAT);
        let empty = OverlayLayer { layer: 1, row_dim: 4, rows: vec![], values: vec![] };
        assert_eq!(from_bytes(&to_bytes(&empty, &BTreeMap::new()).unwrap()).unwrap().0, empty);
    }
}
