//! The digital-page geometry fast path against the full (render + detector)
//! path on every page of a PDF: how many pages take it, text agreement and
//! time.
//!
//! `cargo run --release --example fastpath_cmp -- <models> <book.pdf> [out.json]`
use std::time::Instant;

use kleos_docs::ocr::document::Converter;
use kleos_docs::ocr::Ocr;

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(|w| w.to_string()).collect()
}

/// Similarity of two word sequences: 1 - edit distance / longer length.
fn ratio(a: &[String], b: &[String]) -> f64 {
    let (n, m) = (a.len(), b.len());
    if n.max(m) == 0 {
        return 1.0;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    for i in 1..=n {
        let mut cur = vec![i; m + 1];
        for j in 1..=m {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + if a[i - 1] == b[j - 1] { 0 } else { 1 });
        }
        prev = cur;
    }
    1.0 - prev[m] as f64 / n.max(m) as f64
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ocr = Ocr::load(std::path::Path::new(&a[1]), 1).expect("models");
    let doc = kleos_docs::pdf::Doc::open(std::path::Path::new(&a[2]), None).expect("pdf");
    let n = std::env::var("PAGES").ok().and_then(|v| v.parse().ok()).unwrap_or(doc.pages).min(doc.pages);
    let mut full = Converter::new(&ocr);
    full.fast_digital = false;
    let mut fast = Converter::new(&ocr);
    fast.fast_digital = true;
    let (mut taken, mut same_text, mut same_md, mut ratio_sum) = (0usize, 0usize, 0usize, 0f64);
    let (mut t_full, mut t_fast, mut t_full_on_fast) = (0f64, 0f64, 0f64);
    let mut worst: Vec<(f64, usize)> = vec![];
    for i in 0..n {
        let t0 = Instant::now();
        let pf = full.pdf_page(&doc, i).expect("full");
        let df = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let pq = fast.pdf_page(&doc, i).expect("fast");
        let dq = t0.elapsed().as_secs_f64() * 1000.0;
        t_full += df;
        t_fast += dq;
        if std::env::var("DIFF_PAGE").ok().and_then(|v| v.parse::<usize>().ok()) == Some(i) {
            println!("=== FULL ({} lines)\n{}\n=== FAST ({} lines)\n{}", pf.lines.len(), pf.text(), pq.lines.len(), pq.text());
        }
        if pq.note.contains("geometry fast path") {
            taken += 1;
            t_full_on_fast += df;
            let (wa, wb) = (words(&pf.text()), words(&pq.text()));
            let r = ratio(&wa, &wb);
            ratio_sum += r;
            if wa == wb {
                same_text += 1;
            }
            if pf.markdown == pq.markdown {
                same_md += 1;
            }
            worst.push((r, i));
        }
    }
    worst.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let summary = serde_json::json!({
        "pages": n,
        "fast_pages": taken,
        "same_words": same_text,
        "same_markdown": same_md,
        "mean_word_ratio": if taken > 0 { ratio_sum / taken as f64 } else { 1.0 },
        "worst": worst.iter().take(8).map(|(r, i)| serde_json::json!({"page": i, "ratio": r})).collect::<Vec<_>>(),
        "ms_per_page_full": t_full / n as f64,
        "ms_per_page_with_fast": t_fast / n as f64,
        "ms_per_fast_page_full": if taken > 0 { t_full_on_fast / taken as f64 } else { 0.0 },
    });
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    if let Some(out) = a.get(3) {
        std::fs::write(out, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    }
}
