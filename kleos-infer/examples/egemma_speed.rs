//! EmbeddingGemma v1 vs v2 speed on memory-sized texts: load time, ms per
//! text at batch 1 and 16, weight bytes, peak RSS.
//! Usage: egemma_speed <v1|v2> <precision> <device>
use std::time::Instant;

use kleos_infer::*;

#[repr(C)]
struct Rusage {
    ru_utime: [i64; 2],
    ru_stime: [i64; 2],
    ru_maxrss: i64,
    rest: [i64; 13],
}
extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
}
fn peak_rss_mb() -> f64 {
    let mut ru: Rusage = unsafe { std::mem::zeroed() };
    unsafe { getrusage(0, &mut ru) };
    ru.ru_maxrss as f64 / 1e6
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let engine = EngineConfig {
        device: DevicePref::parse(&a[3])?,
        precision: Precision::parse(&a[2])?,
        ..Default::default()
    };
    let words = ["river", "garden", "Lisbon", "sister", "project", "deadline", "coffee", "train", "violin", "report"];
    let texts: Vec<String> = (0..64)
        .map(|i| format!("I told {} that the {} near the {} was moved to {} last week, and it matters for the {}.", ["Anna", "Tom", "my boss", "Caroline"][i % 4], words[i % 10], words[(i * 3) % 10], words[(i * 7) % 10], words[(i * 9) % 10]))
        .collect();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let t = Instant::now();
    let (emb, info): (Box<dyn TextEmbedder>, Box<dyn Fn() -> Option<(&'static str, usize)>>) = match a[1].as_str() {
        "v1" => {
            let m = GemmaModel::new("google/embeddinggemma-300m", engine)?;
            let e = m.embedder(GEMMA_DOCUMENT_PROMPT, None)?;
            (Box::new(e), Box::new(move || m.loaded_info()))
        }
        _ => {
            let m = Egemma2Model::new(EGEMMA2_REPO, engine)?;
            let e = m.embedder(EGEMMA2_DOCUMENT_PROMPT, None)?;
            (Box::new(e), Box::new(move || m.loaded_info()))
        }
    };
    emb.embed(&["warm up"])?;
    let load = t.elapsed().as_secs_f64();
    for r in refs.iter().take(4) {
        emb.embed(&[r])?;
    }
    let t = Instant::now();
    for r in &refs {
        emb.embed(&[r])?;
    }
    let b1 = t.elapsed().as_secs_f64() * 1000.0 / refs.len() as f64;
    emb.embed(&refs[..16])?;
    let t = Instant::now();
    for c in refs.chunks(16) {
        emb.embed(c)?;
    }
    let b16 = t.elapsed().as_secs_f64() * 1000.0 / refs.len() as f64;
    let (dev, bytes) = info().unwrap_or(("?", 0));
    println!(
        "{} {} {}: load {:.1}s, batch1 {:.1} ms/text, batch16 {:.2} ms/text, weights {:.0} MB, peak RSS {:.0} MB",
        a[1], a[2], dev, load, b1, b16, bytes as f64 / 1e6, peak_rss_mb()
    );
    Ok(())
}
