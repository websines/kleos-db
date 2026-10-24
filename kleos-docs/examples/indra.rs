//! Convert documents with the Rust Indra port and write the same JSON shape
//! as scripts/docs/indra_ref.py, for the parity check.
//!
//!     indra MODEL_DIR SAMPLES_DIR OUT.json [--threads N] [--mode auto|ocr]
use std::time::Instant;

use kleos_docs::ocr::document::{Converter, Mode};
use kleos_docs::ocr::Ocr;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let threads: usize = arg("--threads").map(|s| s.parse().unwrap()).unwrap_or(4);
    let t0 = Instant::now();
    let ocr = Ocr::load(std::path::Path::new(&a[1]), threads).expect("models");
    eprintln!("load {:.2}s", t0.elapsed().as_secs_f64());
    let mut conv = Converter::new(&ocr);
    if arg("--mode").as_deref() == Some("ocr") {
        conv.mode = Mode::Ocr;
    }
    let root = std::path::PathBuf::from(&a[2]);
    let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap()).unwrap();
    let only = arg("--only");
    let mut res = serde_json::Map::new();
    for item in man.as_array().unwrap() {
        let f = item["file"].as_str().unwrap();
        if let Some(o) = &only {
            if !f.contains(o.as_str()) {
                continue;
            }
        }
        let t = Instant::now();
        let mut pages = vec![];
        conv.convert_path(&root.join(f), 0, |i, p| {
            pages.push(serde_json::json!({
                "index": i, "markdown": p.markdown,
                "lines": p.lines.iter().map(|l| serde_json::json!({"box": l.bbox, "text": l.text, "conf": l.conf})).collect::<Vec<_>>(),
                "blocks": p.blocks, "source": p.source, "note": p.note, "rotation": p.rotation, "skew": p.skew, "timings": p.timings,
            }));
        }).unwrap();
        let dt = t.elapsed().as_secs_f64();
        eprintln!("{f} {} {:.2}s", pages.len(), dt);
        res.insert(f.to_string(), serde_json::json!({"pages": pages, "seconds": dt}));
    }
    std::fs::write(&a[3], serde_json::to_string_pretty(&res).unwrap()).unwrap();
}
