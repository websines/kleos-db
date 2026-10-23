//! engram-index parity <tokenizer.json> <ref-dir>...   compare against parity_ref.py dumps
//! engram-index bench  <tokenizer.json> <n-tokens>       hashing throughput (demo config)

use engram_index::{CompressedTokenizer, EngramConfig, NgramHashMapping, Semantics};
use serde_json::Value;
use std::time::Instant;

fn cfg_from_json(c: &Value) -> EngramConfig {
    let u = |v: &Value| v.as_u64().unwrap();
    EngramConfig {
        engram_vocab_size: c["engram_vocab_size"].as_array().unwrap().iter().map(u).collect(),
        max_ngram_size: u(&c["max_ngram_size"]) as usize,
        n_head_per_ngram: u(&c["n_head_per_ngram"]) as usize,
        layer_ids: c["layer_ids"].as_array().unwrap().iter().map(u).collect(),
        pad_id: c["pad_id"].as_u64().map(|p| p as u32),
        seed: u(&c["seed"]),
        semantics: if c["semantics"].as_str() == Some("v41") { Semantics::V41 } else { Semantics::Demo },
    }
}

fn demo_cfg() -> EngramConfig {
    EngramConfig {
        engram_vocab_size: vec![129280 * 5, 129280 * 5],
        max_ngram_size: 3,
        n_head_per_ngram: 8,
        layer_ids: vec![1, 15],
        pad_id: Some(2),
        seed: 0,
        semantics: Semantics::Demo,
    }
}

fn load_ct(path: &str) -> CompressedTokenizer {
    let tok = tokenizers::Tokenizer::from_file(path).expect("tokenizer.json");
    let t = Instant::now();
    let ct = CompressedTokenizer::from_tokenizer(&tok).expect("compress");
    eprintln!("compressed tokenizer: {} -> {} ids in {:.2?}", ct.lookup.len(), ct.num_new_token, t.elapsed());
    ct
}

fn parity(tok_path: &str, dirs: &[String]) -> bool {
    let mut all_ok = true;
    for dir in dirs {
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap()).unwrap();
        let cfg = cfg_from_json(&meta["config"]);
        let mut ok = true;

        // 1. compression table built in Rust from tokenizer.json vs the reference's.
        //    Reported separately: the demo decodes through transformers (version
        //    dependent), V4.1 through the raw Rust tokenizer.
        let built = load_ct(tok_path);
        let bytes = std::fs::read(format!("{dir}/lookup.u32")).unwrap();
        let ref_lookup: Vec<u32> = bytes.chunks(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let lookup_mismatch = ref_lookup.iter().zip(&built.lookup).filter(|(a, b)| a != b).count()
            + ref_lookup.len().abs_diff(built.lookup.len());
        let ref_nnt = meta["num_new_token"].as_u64().unwrap() as u32;
        let lookup_ok = lookup_mismatch == 0 && ref_nnt == built.num_new_token;

        // 2.-3. hashing, given the reference's compression table.
        let hm = NgramHashMapping::new(cfg, CompressedTokenizer::from_lookup(ref_lookup.clone(), ref_nnt));
        ok &= meta["pad_id_compressed"].as_i64() == Some(hm.pad_id);
        if let Some(offs) = meta.get("offsets") {
            for l in &hm.layers {
                let r: Vec<u64> = offs[l.layer_id.to_string()].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
                ok &= r == l.offsets;
            }
        }
        if let Some(ne) = meta.get("num_embeddings") {
            let r: Vec<u64> = ne.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            let ours: Vec<u64> = hm.layers.iter().map(|l| l.total_rows).collect();
            println!("  table rows per layer: ours {ours:?}, config {r:?}");
            ok &= r == ours;
        }

        // 2. multipliers and primes
        let mut param_mismatch = 0;
        for l in &hm.layers {
            let key = l.layer_id.to_string();
            let rm: Vec<i64> = meta["multipliers"][&key].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            let rp: Vec<Vec<u64>> = meta["primes"][&key].as_array().unwrap().iter()
                .map(|h| h.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect()).collect();
            if rm != l.multipliers { param_mismatch += 1; eprintln!("  layer {key} multipliers: ref {rm:?} ours {:?}", l.multipliers); }
            if rp != l.primes { param_mismatch += 1; eprintln!("  layer {key} primes differ"); }
        }
        ok &= param_mismatch == 0;

        // 3. per-token indices for every case and layer
        let (mut cases, mut indices, mut bad) = (0usize, 0usize, 0usize);
        let mut out = Vec::new();
        for line in std::fs::read_to_string(format!("{dir}/cases.jsonl")).unwrap().lines() {
            let c: Value = serde_json::from_str(line).unwrap();
            let ids: Vec<i64> = c["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            for l in &hm.layers {
                hm.hash(l, &ids, &mut out);
                let want: Vec<i64> = c["hashes"][l.layer_id.to_string()].as_array().unwrap().iter()
                    .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_i64().unwrap())).collect();
                indices += want.len();
                if want != out {
                    bad += 1;
                    eprintln!("  case {cases} ({}) layer {} differs", c["kind"], l.layer_id);
                }
            }
            cases += 1;
        }
        ok &= bad == 0;
        println!(
            "{dir}:\n  compression built from tokenizer.json: {} -> {} (reference {} -> {}), {} entries differ -> {}\n  \
             hashing (reference compression): multipliers+primes {} mismatches, {cases} cases x {} layers = {indices} indices, \
             {bad} mismatching (case,layer) -> {}",
            built.lookup.len(), built.num_new_token, ref_lookup.len(), ref_nnt, lookup_mismatch,
            if lookup_ok { "PASS" } else { "DIFF" },
            param_mismatch, hm.layers.len(), if ok { "PASS" } else { "FAIL" }
        );
        all_ok &= ok;
    }
    all_ok
}

fn bench(tok_path: &str, n: usize) {
    let hm = NgramHashMapping::new(demo_cfg(), load_ct(tok_path));
    // Deterministic pseudo-random raw ids (xorshift), no extra deps.
    let mut x = 0x9E3779B97F4A7C15u64;
    let vocab = hm.tokenizer.lookup.len() as u64;
    let ids: Vec<i64> = (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x % vocab) as i64 }).collect();
    let mut out = Vec::new();
    let t = Instant::now();
    let mut sink = 0i64;
    for l in &hm.layers {
        hm.hash(l, &ids, &mut out);
        sink ^= out.iter().fold(0, |a, &b| a ^ b);
    }
    let dt = t.elapsed();
    let per_tok_layer = dt.as_nanos() as f64 / (n * hm.layers.len()) as f64;
    println!(
        "hashed {n} tokens x {} layers ({} indices/token/layer) in {dt:.2?}: {per_tok_layer:.1} ns per token-layer (sink {sink})",
        hm.layers.len(), (hm.cfg.max_ngram_size - 1) * hm.cfg.n_head_per_ngram
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("parity") => std::process::exit(if parity(&args[2], &args[3..]) { 0 } else { 1 }),
        Some("bench") => bench(&args[2], args[3].parse().unwrap()),
        _ => eprintln!("usage: engram-index parity <tokenizer.json> <ref-dir>... | bench <tokenizer.json> <n>"),
    }
}
