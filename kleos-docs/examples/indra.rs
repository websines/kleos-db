//! Convert documents with the Rust Indra port and write the same JSON shape
//! as scripts/docs/indra_ref.py, for the parity check.
//!
//!     indra MODEL_DIR SAMPLES_DIR OUT.json [--threads N] [--mode auto|ocr]
//!     indra MODEL_DIR --pdf BOOK.pdf OUT.json --pages 3,17,40 [--threads N] [--workers W] [--all]
//!
//! The second form mirrors scripts/docs/indra_pages.py: selected pages
//! (lines, Markdown, time), and with --all the whole book on W page workers.
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
    if a[2] == "--pdf" {
        let book = std::path::PathBuf::from(&a[3]);
        let pages: Vec<usize> = arg("--pages").unwrap_or_default().split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
        let doc = kleos_docs::pdf::Doc::open(&book, None).expect("pdf");
        let mut res = serde_json::Map::new();
        let mut out = serde_json::Map::new();
        let t0 = Instant::now();
        for &i in &pages {
            let t = Instant::now();
            let p = conv.pdf_page(&doc, i).expect("page");
            out.insert(i.to_string(), serde_json::json!({"lines": p.lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>(), "markdown": p.markdown, "source": p.source, "s": t.elapsed().as_secs_f64(), "timings": p.timings}));
        }
        let dt = t0.elapsed().as_secs_f64();
        res.insert("load_s".into(), serde_json::json!(0));
        res.insert("sample_s".into(), serde_json::json!(dt));
        res.insert("sample_pages_per_s".into(), serde_json::json!(pages.len() as f64 / dt));
        res.insert("pages".into(), serde_json::Value::Object(out));
        if a.iter().any(|x| x == "--all") {
            let workers: usize = arg("--workers").map(|s| s.parse().unwrap()).unwrap_or(4);
            let t = Instant::now();
            let mut n = 0;
            let mut sink = |_: usize, _: kleos_docs::ocr::Page| n += 1;
            conv.convert_doc_parallel(&doc, 0, doc.pages, workers, &mut sink);
            let dt = t.elapsed().as_secs_f64();
            res.insert("all_pages".into(), serde_json::json!(n));
            res.insert("all_s".into(), serde_json::json!(dt));
            res.insert("all_pages_per_s".into(), serde_json::json!(n as f64 / dt));
            res.insert("all_workers".into(), serde_json::json!(workers));
            eprintln!("all {n} pages {dt:.1}s = {:.3} pages/s on {workers} workers", n as f64 / dt);
        }
        std::fs::write(&a[4], serde_json::to_string_pretty(&res).unwrap()).unwrap();
        eprintln!("sample {} pages {:.1}s", pages.len(), dt);
        return;
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
