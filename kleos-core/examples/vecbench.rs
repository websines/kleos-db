/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Vector-search benchmark harness (kleos upgrade plan item L1).
//!
//! Loads a little-endian `f32` 2-D `.npy` matrix, inserts the first `--n` rows
//! into a relation `vecs { id: Int => bucket: Int, emb: <F32; dim> }` with
//! `bucket = id % 1000`, builds an HNSW index over `emb`, and runs the last
//! `--queries` rows (held out, never inserted) as k-NN queries. Each query runs
//! once per `(selectivity, ef)` pair; a selectivity `s < 1` adds the search
//! filter `bucket < round(s * 1000)`, so `s = 0.01` keeps 1% of the rows.
//!
//! Ground truth is exact brute force in this harness (respecting the filter),
//! cached under `--data-dir` keyed by everything that affects it.
//!
//! Reports build time, recall@k, QPS, p50/p99 latency, the under-fill count
//! (queries that returned fewer than k rows although at least k rows match the
//! filter) and the on-disk size, as a markdown table.
//!
//! ```text
//! cargo run --release -p kleos --example vecbench -- \
//!     --npy .bench-data/dbpedia1536_100k.npy --ef 32,64,128,256 \
//!     --filter-selectivity 1.0,0.1,0.01,0.001
//! ```
//!
//! Reusable for later retrieval work: the index-creation options are a single
//! string (`--index-opts` appends to it), so quantization or other index
//! parameters can be benchmarked without changing the harness.

use kleos::{DataValue, DbInstance, ScriptMutability, Vector};
use ndarray::Array1;
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const BUCKETS: usize = 1000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Metric {
    Cosine,
    L2,
}

impl Metric {
    fn name(self) -> &'static str {
        match self {
            Metric::Cosine => "Cosine",
            Metric::L2 => "L2",
        }
    }
}

struct Args {
    npy: PathBuf,
    data_dir: PathBuf,
    n: Option<usize>,
    queries: usize,
    k: usize,
    ef: Vec<usize>,
    m: usize,
    ef_construction: usize,
    backend: String,
    selectivities: Vec<f64>,
    metric: Metric,
    threads: usize,
    index_opts: String,
    label: String,
    out: Option<PathBuf>,
    reuse_db: bool,
    variants: Vec<String>,
}

fn usage() -> ! {
    eprintln!(
        "usage: vecbench [--npy FILE] [--data-dir DIR] [--n N] [--queries Q] [--k K]\n\
         \x20               [--ef 32,64,..] [--m M] [--ef-construction E] [--backend mem|sqlite]\n\
         \x20               [--filter-selectivity 1.0,0.1,..] [--metric cosine|l2]\n\
         \x20               [--threads T] [--index-opts 'extra: opts'] [--label TEXT]\n\
         \x20               [--out FILE.md] [--reuse-db] [--variants 'K=V K2=V;K=W']\n\
         \n\
         --variants repeats the whole query sweep once per `;`-separated variant,\n\
         setting that variant's space-separated KEY=VALUE environment variables\n\
         first, against the same database (for engine tuning knobs read from the\n\
         environment)."
    );
    std::process::exit(2)
}

fn parse_list<T: std::str::FromStr>(s: &str) -> Vec<T> {
    s.split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| x.trim().parse().unwrap_or_else(|_| usage()))
        .collect()
}

fn parse_args() -> Args {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.bench-data");
    let mut a = Args {
        npy: root.join("dbpedia1536_100k.npy"),
        data_dir: root.join("a1"),
        n: None,
        queries: 1000,
        k: 10,
        ef: vec![32, 64, 128, 256],
        m: 16,
        ef_construction: 128,
        backend: "mem".into(),
        selectivities: vec![1.0],
        metric: Metric::Cosine,
        threads: 4,
        index_opts: String::new(),
        label: String::new(),
        out: None,
        reuse_db: false,
        variants: vec![],
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--npy" => a.npy = val().into(),
            "--data-dir" => a.data_dir = val().into(),
            "--n" => a.n = Some(val().parse().unwrap_or_else(|_| usage())),
            "--queries" => a.queries = val().parse().unwrap_or_else(|_| usage()),
            "--k" => a.k = val().parse().unwrap_or_else(|_| usage()),
            "--ef" => a.ef = parse_list(&val()),
            "--m" => a.m = val().parse().unwrap_or_else(|_| usage()),
            "--ef-construction" => a.ef_construction = val().parse().unwrap_or_else(|_| usage()),
            "--backend" => a.backend = val(),
            "--filter-selectivity" => a.selectivities = parse_list(&val()),
            "--metric" => {
                a.metric = match val().to_ascii_lowercase().as_str() {
                    "cosine" => Metric::Cosine,
                    "l2" => Metric::L2,
                    _ => usage(),
                }
            }
            "--threads" => a.threads = val().parse().unwrap_or_else(|_| usage()),
            "--index-opts" => a.index_opts = val(),
            "--label" => a.label = val(),
            "--out" => a.out = Some(val().into()),
            "--reuse-db" => a.reuse_db = true,
            "--variants" => {
                a.variants = val()
                    .split(';')
                    .map(|v| v.trim().to_string())
                    .collect()
            }
            _ => usage(),
        }
    }
    if !matches!(a.backend.as_str(), "mem" | "sqlite") {
        usage()
    }
    a
}

/// Minimal `.npy` reader: version 1/2/3 headers, `<f4`, C order, 2-D.
fn read_npy(path: &Path) -> (Vec<f32>, usize, usize) {
    let mut f = fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic[..6], b"\x93NUMPY", "not an .npy file");
    let header_len = if magic[6] == 1 {
        let mut b = [0u8; 2];
        f.read_exact(&mut b).unwrap();
        u16::from_le_bytes(b) as usize
    } else {
        let mut b = [0u8; 4];
        f.read_exact(&mut b).unwrap();
        u32::from_le_bytes(b) as usize
    };
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header).unwrap();
    let header = String::from_utf8_lossy(&header);
    assert!(header.contains("'<f4'"), "only little-endian f32 supported: {header}");
    assert!(
        header.contains("'fortran_order': False"),
        "only C order supported: {header}"
    );
    let shape = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .expect("shape");
    let dims: Vec<usize> = shape
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert_eq!(dims.len(), 2, "only 2-D arrays supported");
    let (rows, cols) = (dims[0], dims[1]);
    let mut bytes = vec![0u8; rows * cols * 4];
    f.read_exact(&mut bytes).unwrap();
    let data = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (data, rows, cols)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Exact distance, mirroring the engine's definitions.
fn distance(metric: Metric, a: &[f32], a_norm_sq: f32, b: &[f32], b_norm_sq: f32) -> f64 {
    match metric {
        Metric::Cosine => {
            let den = (a_norm_sq as f64 * b_norm_sq as f64).sqrt();
            if den > 0.0 {
                1.0 - dot(a, b) as f64 / den
            } else {
                2.0
            }
        }
        Metric::L2 => a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>() as f64,
    }
}

/// Ground truth: per query, the ids of the exact k nearest rows passing
/// `bucket < threshold`, nearest first.
fn ground_truth(
    a: &Args,
    data: &[f32],
    dim: usize,
    n: usize,
    queries: &[usize],
    threshold: usize,
) -> Vec<Vec<u32>> {
    let key = format!(
        "gt_{}_n{}_q{}_k{}_{}_t{}.bin",
        a.npy.file_stem().unwrap().to_string_lossy(),
        n,
        queries.len(),
        a.k,
        a.metric.name(),
        threshold
    );
    let path = a.data_dir.join(key);
    if let Ok(bytes) = fs::read(&path) {
        let per = a.k;
        if bytes.len() == queries.len() * per * 4 {
            return bytes
                .chunks_exact(per * 4)
                .map(|c| {
                    c.chunks_exact(4)
                        .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                        .filter(|&id| id != u32::MAX)
                        .collect()
                })
                .collect();
        }
    }
    let t0 = Instant::now();
    let row = |i: usize| &data[i * dim..(i + 1) * dim];
    let norms: Vec<f32> = (0..n).map(|i| dot(row(i), row(i))).collect();
    let mut out: Vec<Vec<u32>> = vec![vec![]; queries.len()];
    let chunk = queries.len().div_ceil(a.threads.max(1));
    std::thread::scope(|s| {
        for (qs, outs) in queries.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let norms = &norms;
            s.spawn(move || {
                for (&qi, o) in qs.iter().zip(outs.iter_mut()) {
                    let q = row(qi);
                    let qn = dot(q, q);
                    let mut best: Vec<(f64, u32)> = Vec::with_capacity(a.k + 1);
                    for id in 0..n {
                        if id % BUCKETS >= threshold {
                            continue;
                        }
                        let d = distance(a.metric, q, qn, row(id), norms[id]);
                        if best.len() < a.k || d < best[best.len() - 1].0 {
                            let pos = best.partition_point(|x| x.0 <= d);
                            best.insert(pos, (d, id as u32));
                            best.truncate(a.k);
                        }
                    }
                    *o = best.into_iter().map(|x| x.1).collect();
                }
            });
        }
    });
    let mut bytes = Vec::with_capacity(queries.len() * a.k * 4);
    for ids in &out {
        for i in 0..a.k {
            bytes.extend_from_slice(&ids.get(i).copied().unwrap_or(u32::MAX).to_le_bytes());
        }
    }
    fs::write(&path, bytes).unwrap();
    eprintln!(
        "ground truth (threshold {threshold}) computed in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    out
}

fn to_vec(data: &[f32], dim: usize, i: usize) -> DataValue {
    DataValue::Vec(Vector::F32(Array1::from(data[i * dim..(i + 1) * dim].to_vec())))
}

fn dir_size(p: &Path) -> u64 {
    let mut total = 0;
    let parent = p.parent().unwrap();
    let stem = p.file_name().unwrap().to_string_lossy().to_string();
    if let Ok(rd) = fs::read_dir(parent) {
        for e in rd.flatten() {
            // the database file plus its -wal / -shm / -journal siblings
            if e.file_name().to_string_lossy().starts_with(&stem) {
                total += e.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

/// CPU time consumed by this process. Wall-clock latency is the headline
/// number, but on a shared, overloaded machine it mostly measures scheduling;
/// CPU time is far steadier for before/after comparisons. (Process-wide rather
/// than per-thread, because the engine runs rule evaluation on a rayon pool.)
fn process_cpu_time() -> Duration {
    #[repr(C)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }
    extern "C" {
        fn clock_gettime(clk: i32, tp: *mut Timespec) -> i32;
    }
    #[cfg(target_os = "macos")]
    const CLOCK_PROCESS_CPUTIME_ID: i32 = 12;
    #[cfg(not(target_os = "macos"))]
    const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: plain libc call writing into a properly laid out timespec
    // (64-bit targets).
    let rc = unsafe { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    if rc != 0 {
        return Duration::ZERO;
    }
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn main() {
    let a = parse_args();
    fs::create_dir_all(&a.data_dir).expect("data dir");
    let t_load = Instant::now();
    let (data, rows, dim) = read_npy(&a.npy);
    eprintln!(
        "loaded {rows}x{dim} from {:?} in {:.1}s",
        a.npy,
        t_load.elapsed().as_secs_f64()
    );
    assert!(a.queries < rows, "--queries must leave rows to insert");
    let n = a.n.unwrap_or(rows - a.queries).min(rows - a.queries);
    let queries: Vec<usize> = (rows - a.queries..rows).collect();

    // ---- build -------------------------------------------------------------
    let db_path = a.data_dir.join(format!("vecbench_n{n}_m{}.db", a.m));
    let reuse = a.backend == "sqlite" && a.reuse_db && db_path.exists();
    if a.backend == "sqlite" && !reuse {
        for suffix in ["", "-wal", "-shm", "-journal"] {
            let _ = fs::remove_file(format!("{}{suffix}", db_path.display()));
        }
    }
    let db = DbInstance::new(&a.backend, &db_path, "").expect("open db");
    let mut insert_time = Duration::ZERO;
    let mut build_time = Duration::ZERO;
    if !reuse {
        db.run_script(
            &format!(":create vecs {{ id: Int => bucket: Int, emb: <F32; {dim}> }}"),
            BTreeMap::new(),
            ScriptMutability::Mutable,
        )
        .expect("create");
        let t = Instant::now();
        let batch = 5000;
        for start in (0..n).step_by(batch) {
            let rows: Vec<DataValue> = (start..(start + batch).min(n))
                .map(|i| {
                    DataValue::List(vec![
                        DataValue::from(i as i64),
                        DataValue::from((i % BUCKETS) as i64),
                        to_vec(&data, dim, i),
                    ])
                })
                .collect();
            let params = BTreeMap::from([("rows".to_string(), DataValue::List(rows))]);
            db.run_script(
                "?[id, bucket, emb] <- $rows :put vecs { id => bucket, emb }",
                params,
                ScriptMutability::Mutable,
            )
            .expect("insert");
        }
        insert_time = t.elapsed();
        let mut opts = format!(
            "dim: {dim}, m: {}, dtype: F32, fields: [emb], distance: {}, ef_construction: {}",
            a.m,
            a.metric.name(),
            a.ef_construction
        );
        if !a.index_opts.is_empty() {
            opts.push_str(", ");
            opts.push_str(&a.index_opts);
        }
        let t = Instant::now();
        db.run_script(
            &format!("::hnsw create vecs:idx {{ {opts} }}"),
            BTreeMap::new(),
            ScriptMutability::Mutable,
        )
        .expect("index build");
        build_time = t.elapsed();
        eprintln!(
            "inserted {n} rows in {:.1}s, built index in {:.1}s",
            insert_time.as_secs_f64(),
            build_time.as_secs_f64()
        );
    }
    let disk = if a.backend == "sqlite" {
        format!("{:.1} MiB", dir_size(&db_path) as f64 / (1024.0 * 1024.0))
    } else {
        "-".to_string()
    };

    let run_query = |qi: usize, ef: usize, threshold: usize| -> Vec<i64> {
        let filter = if threshold >= BUCKETS {
            String::new()
        } else {
            format!(", filter: bucket < {threshold}")
        };
        let script = format!(
            "?[id, dist] := ~vecs:idx{{ id, bucket | query: $q, k: {}, ef: {ef}, \
             bind_distance: dist{filter} }}",
            a.k
        );
        let params = BTreeMap::from([("q".to_string(), to_vec(&data, dim, qi))]);
        let res = db
            .run_script(&script, params, ScriptMutability::Immutable)
            .expect("query");
        res.rows.iter().map(|r| r[0].get_int().unwrap()).collect()
    };

    // warm caches once
    for &qi in queries.iter().take(100) {
        run_query(qi, a.ef[0], BUCKETS);
    }

    // ---- query sweep -------------------------------------------------------
    let mut table = String::new();
    let label = if a.label.is_empty() {
        String::new()
    } else {
        format!(" — {}", a.label)
    };
    table.push_str(&format!(
        "\n### vecbench{label}\n\n{} rows x {dim}d, {} held-out queries, k={}, M={}, \
         ef_construction={}, {}, backend {}; insert {:.1}s, index build {:.1}s, disk {disk}\n\n",
        n,
        queries.len(),
        a.k,
        a.m,
        a.ef_construction,
        a.metric.name(),
        a.backend,
        insert_time.as_secs_f64(),
        build_time.as_secs_f64(),
    ));
    table.push_str(
        "| variant | selectivity | matches | ef | recall@k | QPS | p50 ms | p99 ms | CPU ms/q | under-fill |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    let variants = if a.variants.is_empty() {
        vec![String::new()]
    } else {
        a.variants.clone()
    };
    for variant in &variants {
    for kv in variant.split_whitespace() {
        if let Some((k, v)) = kv.split_once('=') {
            std::env::set_var(k, v);
        }
    }
    let vname = if variant.is_empty() { "-" } else { variant.as_str() };
    for &s in &a.selectivities {
        let threshold = ((s * BUCKETS as f64).round() as usize).clamp(1, BUCKETS);
        let matches = (0..n).filter(|i| i % BUCKETS < threshold).count();
        let gt = ground_truth(&a, &data, dim, n, &queries, threshold);
        for &ef in &a.ef {
            let mut lat = Vec::with_capacity(queries.len());
            let mut hits = 0usize;
            let mut expected = 0usize;
            let mut under = 0usize;
            let t = Instant::now();
            let cpu0 = process_cpu_time();
            for (qi, truth) in queries.iter().zip(&gt) {
                let tq = Instant::now();
                let got = run_query(*qi, ef, threshold);
                lat.push(tq.elapsed());
                for id in &got {
                    assert!(
                        (*id as usize) % BUCKETS < threshold,
                        "filter violated: id {id}"
                    );
                }
                if got.len() < a.k && matches >= a.k {
                    under += 1;
                }
                expected += truth.len();
                hits += truth
                    .iter()
                    .filter(|id| got.contains(&(**id as i64)))
                    .count();
            }
            let total = t.elapsed();
            let cpu = process_cpu_time().saturating_sub(cpu0);
            lat.sort();
            let line = format!(
                "| {vname} | {s} | {matches} | {ef} | {:.4} | {:.1} | {:.3} | {:.3} | {:.3} | {under} |\n",
                hits as f64 / expected.max(1) as f64,
                queries.len() as f64 / total.as_secs_f64(),
                percentile(&lat, 0.5).as_secs_f64() * 1e3,
                percentile(&lat, 0.99).as_secs_f64() * 1e3,
                cpu.as_secs_f64() * 1e3 / queries.len() as f64,
            );
            eprint!("{line}");
            table.push_str(&line);
        }
    }
    }
    println!("{table}");
    if let Some(out) = &a.out {
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(out)
            .expect("open --out");
        f.write_all(table.as_bytes()).unwrap();
    }
}
