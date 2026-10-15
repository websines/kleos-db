//! Embed texts from a JSONL file (one string per line) and write vectors as
//! little-endian f32 to an output file; prints throughput and peak RSS.
//! Usage: embed_bench <model: gemma|potion-8M|potion-32M> <precision> <device> <in.jsonl> <out.f32> [prompt: query|document] [max_batch_tokens]
use std::io::{BufRead, Write};
use std::time::Instant;

use kleos_infer::*;

fn peak_rss_mb() -> f64 {
    let mut ru: libc_rusage = unsafe { std::mem::zeroed() };
    unsafe { getrusage(0, &mut ru) };
    ru.ru_maxrss as f64 / 1e6
}
#[repr(C)]
struct libc_rusage {
    ru_utime: [i64; 2],
    ru_stime: [i64; 2],
    ru_maxrss: i64,
    rest: [i64; 13],
}
extern "C" {
    fn getrusage(who: i32, usage: *mut libc_rusage) -> i32;
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let texts: Vec<String> = std::io::BufReader::new(std::fs::File::open(&a[4])?)
        .lines()
        .map(|l| serde_json::from_str::<String>(&l.unwrap()).unwrap())
        .collect();
    let engine = EngineConfig {
        device: DevicePref::parse(&a[3])?,
        precision: Precision::parse(&a[2])?,
        max_batch_tokens: a.get(7).map(|s| s.parse().unwrap()).unwrap_or(8192),
        ..Default::default()
    };
    let prompt = match a.get(6).map(|s| s.as_str()) {
        Some("query") => GEMMA_QUERY_PROMPT,
        _ => GEMMA_DOCUMENT_PROMPT,
    };
    let emb: Box<dyn TextEmbedder> = match a[1].as_str() {
        "gemma" => Box::new(GemmaModel::new("google/embeddinggemma-300m", engine)?.embedder(prompt, None)?),
        "potion-8M" => Box::new(StaticEmbedder::new("minishlab/potion-base-8M")?),
        "potion-32M" => Box::new(StaticEmbedder::new("minishlab/potion-base-32M")?),
        m => panic!("unknown model {m}"),
    };
    let t = Instant::now();
    emb.embed(&["warm up"])?;
    let load = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[5])?);
    let chunk = 256;
    for c in texts.chunks(chunk) {
        let refs: Vec<&str> = c.iter().map(|s| s.as_str()).collect();
        for v in emb.embed(&refs)? {
            for x in v {
                out.write_all(&x.to_le_bytes())?;
            }
        }
    }
    let secs = t.elapsed().as_secs_f64();
    println!(
        "{{\"model\":\"{}\",\"n\":{},\"load_s\":{:.2},\"secs\":{:.2},\"texts_per_s\":{:.1},\"peak_rss_mb\":{:.0},\"dim\":{}}}",
        emb.model_id(), texts.len(), load, secs, texts.len() as f64 / secs, peak_rss_mb(), emb.dim()
    );
    Ok(())
}
