//! Does `GemmaModel::unload` (and dropping the model) give memory back?
//! Ignored by default: needs google/embeddinggemma-300m in the HF cache.
//! Prints the process footprint per load/unload round. On macOS the system
//! allocator keeps freed large blocks (the load-time f32 buffers, and on CPU
//! the weights) in its large cache; run with `MallocLargeCache=0` to see the
//! memory the model itself holds (measured, Metal int8: 432 MB loaded, 11 MB
//! after unload, flat across rounds; before the autorelease-pool fix in
//! `gemma.rs`, debug builds kept ~315 MB per round).
//! `KLEOS_DEVICE=cpu|metal cargo +1.99.0 test -p kleos-infer --features metal \
//!   --test unload_memory -- --ignored --nocapture`

use kleos_infer::{
    DevicePref, EngineConfig, GemmaModel, Precision, TextEmbedder, GEMMA_DOCUMENT_PROMPT,
};

fn footprint_mb() -> f64 {
    let out = std::process::Command::new("footprint")
        .args(["-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().find(|l| l.contains("Footprint:")).unwrap_or("");
    let v: Vec<&str> = line.split("Footprint:").nth(1).unwrap_or("").split_whitespace().collect();
    let n: f64 = v.first().and_then(|x| x.parse().ok()).unwrap_or(0.0);
    match v.get(1).copied() {
        Some("GB") => n * 1024.0,
        Some("KB") => n / 1024.0,
        _ => n,
    }
}

#[test]
#[ignore]
fn unload_frees_memory() {
    let device = DevicePref::parse(&std::env::var("KLEOS_DEVICE").unwrap_or("auto".into())).unwrap();
    let engine = EngineConfig {
        device,
        precision: Precision::Q8,
        ..Default::default()
    };
    eprintln!("start: {:.0} MB", footprint_mb());
    for round in 0..3 {
        let model = GemmaModel::new("google/embeddinggemma-300m", engine.clone()).unwrap();
        let e = model.embedder(GEMMA_DOCUMENT_PROMPT, None).unwrap();
        e.embed(&["hello world", "kleos is a database"]).unwrap();
        eprintln!("round {round} loaded {:?}: {:.0} MB", model.loaded_info(), footprint_mb());
        model.unload();
        std::thread::sleep(std::time::Duration::from_millis(500));
        eprintln!("round {round} unloaded: {:.0} MB", footprint_mb());
        drop(e);
        drop(model);
        std::thread::sleep(std::time::Duration::from_millis(500));
        eprintln!("round {round} dropped: {:.0} MB", footprint_mb());
    }
}
