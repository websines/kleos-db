/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Parity against the reference dumps of the research spike
//! (docs/research/engram.md §5). The dumps (tens of MB) live outside git, in
//! `$KLEOS_ENGRAM_BENCH` or `.bench-data/engram` next to the checkout; each
//! test skips with a note when its inputs are absent.

#![cfg(feature = "tokenizer")]

use std::path::{Path, PathBuf};

use kleos_engram::{parity, CompressedTokenizer, EngramSpec, SpecOverrides};

fn bench_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("KLEOS_ENGRAM_BENCH") {
        return Some(PathBuf::from(d));
    }
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    // main checkout (<root>/kleos/kleos-engram) or a worktree (<root>/.worktrees/x/kleos-engram)
    [here.join("../../.bench-data/engram"), here.join("../../../.bench-data/engram")]
        .into_iter()
        .find(|p| p.join("ref").is_dir())
}

fn gpt2_tokenizer() -> Option<PathBuf> {
    let home = std::env::var("HF_HOME").ok().map(PathBuf::from)?;
    let snaps = home.join("hub/models--openai-community--gpt2/snapshots");
    std::fs::read_dir(snaps).ok()?.flatten().map(|e| e.path().join("tokenizer.json")).find(|p| p.is_file())
}

fn check(tokenizer: &Path, dirs: &[PathBuf], expect_compression: bool) {
    let tok = tokenizers::Tokenizer::from_file(tokenizer).unwrap();
    let built = CompressedTokenizer::from_tokenizer(&tok).unwrap();
    for d in dirs {
        let r = parity::check_dir(d, Some(&built)).unwrap();
        eprintln!("{}: {r:?}", d.display());
        assert!(r.hashing_ok(), "{}: hashing parity failed: {r:?}", d.display());
        if expect_compression {
            assert!(r.compression_ok(), "{}: compression differs: {r:?}", d.display());
        }
    }
}

#[test]
fn v41_flash_reference() {
    let Some(b) = bench_dir() else { return eprintln!("skip: no .bench-data/engram") };
    let (tok, dir) = (b.join("tok-v41/tokenizer.json"), b.join("ref/v41"));
    if !tok.is_file() || !dir.is_dir() {
        return eprintln!("skip: V4.1 refs absent");
    }
    check(&tok, &[dir], true);
}

#[test]
fn demo_references() {
    let Some(b) = bench_dir() else { return eprintln!("skip: no .bench-data/engram") };
    let tok = b.join("tok-v3/tokenizer.json");
    let dirs: Vec<PathBuf> = ["demo", "paper27b", "ngram4"].iter().map(|d| b.join("ref").join(d)).filter(|d| d.is_dir()).collect();
    if !tok.is_file() || dirs.is_empty() {
        return eprintln!("skip: demo refs absent");
    }
    // the demo built its table through transformers 5.x (a different decoder),
    // so only hashing is compared (docs/research/engram.md §5 note)
    check(&tok, &dirs, false);
}

#[test]
fn nanogpt_reference_and_spec_from_config() {
    let Some(b) = bench_dir() else { return eprintln!("skip: no .bench-data/engram") };
    let dir = b.join("overlay/ref-nanogpt");
    let cfg = b.join("models/engram-nanogpt/engram-seed1337/config.json");
    let Some(tok) = gpt2_tokenizer() else { return eprintln!("skip: no GPT-2 tokenizer in $HF_HOME") };
    if !dir.is_dir() || !cfg.is_file() {
        return eprintln!("skip: nanogpt refs absent");
    }
    check(&tok, &[dir.clone()], true);
    // the production loader on the model's own config.json gives the same table
    let spec = EngramSpec::from_files(&cfg, &tok, &SpecOverrides { pad_id: Some(50256), ..Default::default() }).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let l = spec.mapping.layer(1).unwrap();
    assert_eq!(serde_json::json!(l.offsets), meta["offsets"]["1"]);
    assert_eq!(serde_json::json!(l.multipliers), meta["multipliers"]["1"]);
    assert_eq!(spec.row_dim, 64);
    let first = std::fs::read_to_string(dir.join("cases.jsonl")).unwrap();
    let c: serde_json::Value = serde_json::from_str(first.lines().next().unwrap()).unwrap();
    let ids: Vec<i64> = c["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    let rows: Vec<i64> = spec.rows_for_ids(&ids).iter().map(|h| h.row).collect();
    let want: Vec<i64> = c["hashes"]["1"].as_array().unwrap().iter().flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_i64().unwrap())).collect();
    assert_eq!(rows, want);
}
