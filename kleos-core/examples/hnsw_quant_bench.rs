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
//! [BENCH_F16=1] [BENCH_ENGINE=sqlite] cargo run --release -p kleos \
//!     --example hnsw_quant_bench -- <data.npy> <work dir> \
//!     [quantize list, e.g. none,turbo4,binary] [ef list, e.g. 64,128] [n queries]
//! ```
//!
//! The last `n queries` rows are held out as queries; the rest are loaded
//! into one relation that gets one cosine index per quantize setting
//! (`BENCH_F16=1` adds an `<F16; d>` copy with an unquantized index). Ground
//! truth is brute force, cached next to the data. Queries run one at a time,
//! interleaving the indexes per query; wall latency and process CPU time are
//! reported per index and `ef`.

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

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}
extern "C" {
    fn clock_gettime(clk: i32, tp: *mut Timespec) -> i32;
}

/// Process CPU time in ms (all threads), so per-query cost is comparable
/// on a shared, oversubscribed machine.
fn cpu_ms() -> f64 {
    #[cfg(target_os = "macos")]
    const CLOCK_PROCESS_CPUTIME_ID: i32 = 12;
    #[cfg(not(target_os = "macos"))]
    const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 * 1e3 + ts.tv_nsec as f64 / 1e6
}

fn import(db: &DbInstance, rel: &str, base: &[f32], d: usize) {
    let nb = base.len() / d;
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
            rel.to_string(),
            NamedRows::new(vec!["id".into(), "emb".into()], rows),
        );
        db.import_relations(data).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_path = &args[1];
    let work = Path::new(&args[2]);
    let quants: Vec<String> = args
        .get(3)
        .map(|s| s.split(',').map(String::from).collect())
        .unwrap_or(vec!["none".into(), "turbo4".into()]);
    let efs: Vec<usize> = args
        .get(4)
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![64, 128]);
    let nq: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let with_f16 = std::env::var_os("BENCH_F16").is_some();
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

    let db_dir = work.join(format!("{engine}-bench"));
    let _ = std::fs::remove_dir_all(&db_dir);
    std::fs::create_dir_all(&db_dir).unwrap();
    let db_path = db_dir.join("db");
    let db = DbInstance::new(&engine, db_path.to_str().unwrap(), "").unwrap();
    let run = |s: &str| {
        db.run_script(s, BTreeMap::new(), ScriptMutability::Mutable)
            .unwrap_or_else(|e| panic!("{e:?}\n{s}"))
    };
    let mib = |b: u64| b as f64 / 1048576.0;

    run(&format!(":create v {{ id: Int => emb: <F32; {d}> }}"));
    let t = Instant::now();
    import(&db, "v", base, d);
    let mut size = dir_size(&db_dir);
    println!(
        "base relation <F32; {d}>: {nb} rows, load {:.1} s, {:.0} MiB",
        t.elapsed().as_secs_f64(),
        mib(size)
    );
    if with_f16 {
        run(&format!(":create vh {{ id: Int => emb: <F16; {d}> }}"));
        let t = Instant::now();
        import(&db, "vh", base, d);
        let s2 = dir_size(&db_dir);
        println!(
            "base relation <F16; {d}>: {nb} rows, load {:.1} s, {:.0} MiB",
            t.elapsed().as_secs_f64(),
            mib(s2 - size)
        );
        size = s2;
    }

    // (relation, index name, label)
    let mut indexes: Vec<(String, String, String)> = vec![];
    println!("\n| index | build s (wall) | build CPU s | index MiB |\n|---|---|---|---|");
    let mut builds: Vec<(&str, &str, String)> =
        quants.iter().map(|q| ("v", "F32", q.clone())).collect();
    if with_f16 {
        builds.push(("vh", "F16", "none".into()));
    }
    for (rel, dtype, q) in builds {
        let name = format!("i_{q}");
        let t = Instant::now();
        let c = cpu_ms();
        run(&format!(
            "::hnsw create {rel}:{name} {{ dim: {d}, m: 16, dtype: {dtype}, fields: [emb], \
             distance: Cosine, ef_construction: 100, quantize: {q} }}"
        ));
        let wall = t.elapsed().as_secs_f64();
        let cpu = (cpu_ms() - c) / 1e3;
        let s2 = dir_size(&db_dir);
        let label = format!("{dtype} {q}");
        println!("| {label} | {wall:.1} | {cpu:.1} | {:.0} |", mib(s2.saturating_sub(size)));
        size = s2;
        indexes.push((rel.to_string(), name, label));
    }

    let qparam = |qi: usize| {
        let mut p = BTreeMap::new();
        p.insert(
            "q".to_string(),
            DataValue::Vec(Vector::F32(ndarray::Array1::from(
                queries[qi * d..(qi + 1) * d].to_vec(),
            ))),
        );
        p
    };
    println!(
        "\n| index | ef | recall@10 | QPS (wall) | p50 ms | p99 ms | CPU ms/query |\n|---|---|---|---|---|---|---|"
    );
    for &ef in &efs {
        let scripts: Vec<String> = indexes
            .iter()
            .map(|(rel, name, _)| {
                format!("?[id] := ~{rel}:{name}{{id | query: $q, k: {k}, ef: {ef}}}")
            })
            .collect();
        // Warm-up.
        for qi in 0..nq.min(100) {
            for s in &scripts {
                db.run_script(s, qparam(qi), ScriptMutability::Immutable).unwrap();
            }
        }
        let m = scripts.len();
        let mut lat = vec![vec![]; m];
        let mut cpu = vec![0.0f64; m];
        let mut hits = vec![0usize; m];
        // Interleave indexes per query (rotating order) so load swings on a
        // shared machine hit every index alike.
        for qi in 0..nq {
            for j in 0..m {
                let ix = (qi + j) % m;
                let p = qparam(qi);
                let c = cpu_ms();
                let t = Instant::now();
                let res = db.run_script(&scripts[ix], p, ScriptMutability::Immutable).unwrap();
                lat[ix].push(t.elapsed().as_secs_f64() * 1e3);
                cpu[ix] += cpu_ms() - c;
                hits[ix] += res
                    .rows
                    .iter()
                    .filter(|r| gt[qi].contains(&r[0].get_int().unwrap()))
                    .count();
            }
        }
        for ix in 0..m {
            let l = &mut lat[ix];
            let mean = l.iter().sum::<f64>() / l.len() as f64;
            l.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |p: f64| l[((l.len() as f64 * p) as usize).min(l.len() - 1)];
            println!(
                "| {} | {ef} | {:.4} | {:.1} | {:.2} | {:.2} | {:.2} |",
                indexes[ix].2,
                hits[ix] as f64 / (nq * k) as f64,
                1e3 / mean,
                pct(0.5),
                pct(0.99),
                cpu[ix] / nq as f64
            );
        }
    }
    drop(db);
    if std::env::var_os("BENCH_KEEP").is_none() {
        let _ = std::fs::remove_dir_all(&db_dir);
    }
}
