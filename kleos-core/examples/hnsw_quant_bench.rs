/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! End-to-end HNSW benchmark for quantized traversal (A2) and half-width
//! columns (A3) on a real embedding set (`.npy`, float32, row-major).
//!
//! ```text
//! cargo run --release -p kleos --example hnsw_quant_bench -- \
//!     <data.npy> <work dir> <quantize: none|binary|turbo2|turbo4|turbo8> \
//!     [eltype F32|F16] [ef list, e.g. 64,128] [n queries]
//! ```
//!
//! The last `n queries` rows are held out as queries; the rest are indexed
//! (cosine). Ground truth is brute force, cached next to the data. Queries
//! run one at a time on one thread. Prints one Markdown table row per `ef`.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use kleos::{DataValue, DbInstance, NamedRows, ScriptMutability, Vector};

fn load_npy(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).expect("read npy");
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
    let shape = header.split("'shape': (").nth(1).unwrap();
    let dims: Vec<usize> = shape
        .split(')')
        .next()
        .unwrap()
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let v = bytes[10 + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (v, dims[0], dims[1])
}

fn dir_size(p: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let m = e.metadata().unwrap();
            total += if m.is_dir() { dir_size(&e.path()) } else { m.len() };
        }
    }
    total
}

fn ground_truth(base: &[f32], queries: &[f32], d: usize, k: usize, cache: &Path) -> Vec<Vec<i64>> {
    let nq = queries.len() / d;
    if let Ok(s) = std::fs::read_to_string(cache) {
        let gt: Vec<Vec<i64>> = s
            .lines()
            .map(|l| l.split(',').map(|x| x.parse().unwrap()).collect())
            .collect();
        if gt.len() == nq {
            return gt;
        }
    }
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = base.len() / d;
    let norms: Vec<f32> = (0..nb).map(|i| norm(&base[i * d..(i + 1) * d])).collect();
    let threads = 4;
    let chunk = nq.div_ceil(threads);
    let gt: Vec<Vec<i64>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let norms = &norms;
                s.spawn(move || {
                    (t * chunk..((t + 1) * chunk).min(nq))
                        .map(|qi| {
                            let q = &queries[qi * d..(qi + 1) * d];
                            let mut sc: Vec<(f32, i64)> = (0..nb)
                                .map(|i| {
                                    let b = &base[i * d..(i + 1) * d];
                                    let dot: f32 = q.iter().zip(b).map(|(x, y)| x * y).sum();
                                    (dot / norms[i], i as i64)
                                })
                                .collect();
                            sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                            sc[..k].iter().map(|x| x.1).collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let text: Vec<String> = gt
        .iter()
        .map(|r| r.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
        .collect();
    let _ = std::fs::write(cache, text.join("\n"));
    gt
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_path = &args[1];
    let work = Path::new(&args[2]);
    let quant = args.get(3).map(String::as_str).unwrap_or("none");
    let eltype = args.get(4).map(String::as_str).unwrap_or("F32");
    let efs: Vec<usize> = args
        .get(5)
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![64, 128]);
    let nq: usize = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let engine = std::env::var("BENCH_ENGINE").unwrap_or_else(|_| "sqlite".into());
    let k = 10;

    let (all, n, d) = load_npy(data_path);
    let nb = n - nq;
    let (base, queries) = all.split_at(nb * d);
    let gt = ground_truth(
        base,
        queries,
        d,
        k,
        &Path::new(data_path).with_extension(format!("gt{nq}.csv")),
    );

    let db_dir = work.join(format!("{engine}-{quant}-{eltype}"));
    let _ = std::fs::remove_dir_all(&db_dir);
    std::fs::create_dir_all(&db_dir).unwrap();
    let db_path = db_dir.join("db");
    let db = DbInstance::new(&engine, db_path.to_str().unwrap(), "").unwrap();
    let run = |s: &str| {
        db.run_script(s, BTreeMap::new(), ScriptMutability::Mutable)
            .unwrap_or_else(|e| panic!("{e:?}\n{s}"))
    };
    run(&format!(":create v {{ id: Int => emb: <{eltype}; {d}> }}"));

    let t = Instant::now();
    for start in (0..nb).step_by(5000) {
        let end = (start + 5000).min(nb);
        let rows = (start..end)
            .map(|i| {
                vec![
                    DataValue::from(i as i64),
                    DataValue::Vec(Vector::F32(ndarray::Array1::from(
                        base[i * d..(i + 1) * d].to_vec(),
                    ))),
                ]
            })
            .collect();
        let mut data = BTreeMap::new();
        data.insert(
            "v".to_string(),
            NamedRows::new(vec!["id".into(), "emb".into()], rows),
        );
        db.import_relations(data).unwrap();
    }
    let load_s = t.elapsed().as_secs_f64();
    let size_data = dir_size(&db_dir);

    let t = Instant::now();
    run(&format!(
        "::hnsw create v:idx {{ dim: {d}, m: 16, dtype: {eltype}, fields: [emb], \
         distance: Cosine, ef_construction: 100, quantize: {quant} }}"
    ));
    let build_s = t.elapsed().as_secs_f64();
    let size_total = dir_size(&db_dir);

    for &ef in &efs {
        let script = format!("?[id] := ~v:idx{{id | query: $q, k: {k}, ef: {ef}}}");
        // Warm-up pass over a slice of the queries.
        for qi in 0..nq.min(100) {
            let mut p = BTreeMap::new();
            p.insert(
                "q".to_string(),
                DataValue::Vec(Vector::F32(ndarray::Array1::from(
                    queries[qi * d..(qi + 1) * d].to_vec(),
                ))),
            );
            db.run_script(&script, p, ScriptMutability::Immutable).unwrap();
        }
        let mut lat = Vec::with_capacity(nq);
        let mut hits = 0usize;
        let wall = Instant::now();
        for qi in 0..nq {
            let mut p = BTreeMap::new();
            p.insert(
                "q".to_string(),
                DataValue::Vec(Vector::F32(ndarray::Array1::from(
                    queries[qi * d..(qi + 1) * d].to_vec(),
                ))),
            );
            let t = Instant::now();
            let res = db.run_script(&script, p, ScriptMutability::Immutable).unwrap();
            lat.push(t.elapsed().as_secs_f64() * 1e3);
            hits += res
                .rows
                .iter()
                .filter(|r| gt[qi].contains(&r[0].get_int().unwrap()))
                .count();
        }
        let wall_s = wall.elapsed().as_secs_f64();
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
        println!(
            "| {engine} | {eltype} | {quant} | {ef} | {load_s:.1} | {build_s:.1} | {:.0} | {:.0} | {:.4} | {:.0} | {:.2} | {:.2} |",
            size_data as f64 / 1048576.0,
            (size_total - size_data.min(size_total)) as f64 / 1048576.0,
            hits as f64 / (nq * k) as f64,
            nq as f64 / wall_s,
            pct(0.5),
            pct(0.99),
        );
    }
    drop(db);
    if std::env::var_os("BENCH_KEEP").is_none() {
        let _ = std::fs::remove_dir_all(&db_dir);
    }
}
