//! Orientation preview and skew search timings on rendered PDF pages (the
//! per-page image ops before the orientation model): the minimum of `reps`
//! runs per page, so other load on the machine mostly drops out.
//!
//!     skew_bench BOOK.pdf --pages 20,60,120 [--reps 7]
use std::time::Instant;

use kleos_docs::ocr::pipeline::{estimate_skew, preview};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let pages: Vec<usize> = arg("--pages").unwrap_or_default().split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
    let reps: usize = arg("--reps").map(|s| s.parse().unwrap()).unwrap_or(7);
    let doc = kleos_docs::pdf::Doc::open(std::path::Path::new(&a[1]), None).expect("pdf");
    let (mut tp, mut ts) = (0.0, 0.0);
    for &i in &pages {
        let g = doc.with_page(i, |p| p.render_gray(200.0 / 72.0)).expect("page");
        let (mut bp, mut bs) = (f64::INFINITY, f64::INFINITY);
        let mut angle = 0.0;
        let mut bt = f64::INFINITY;
        for _ in 0..reps {
            let t0 = Instant::now();
            let small = preview(&g, 1600);
            let t1 = Instant::now();
            angle = estimate_skew(&small, 800, 1);
            let t2 = Instant::now();
            bp = bp.min((t1 - t0).as_secs_f64() * 1000.0);
            // the two ink thumbnails estimate_skew makes first (800 and 400 px)
            let t3 = Instant::now();
            for side in [800usize, 400] {
                let sc = side as f64 / small.h.max(small.w) as f64;
                let (w, h) = (((small.w as f64 * sc).round_ties_even() as usize).max(1), ((small.h as f64 * sc).round_ties_even() as usize).max(1));
                std::hint::black_box(kleos_docs::ocr::cv::resize_area(&small, w, h).inverted());
            }
            bt = bt.min(t3.elapsed().as_secs_f64() * 1000.0);
            bs = bs.min((t2 - t1).as_secs_f64() * 1000.0);
        }
        println!("page {i} ({}x{}): preview {bp:.1} ms, skew {bs:.1} ms (thumbnails {bt:.1} ms), angle {angle:.2}", g.w, g.h);
        tp += bp;
        ts += bs;
    }
    let n = pages.len().max(1) as f64;
    println!("mean: preview {:.1} ms, skew {:.1} ms, total {:.1} ms", tp / n, ts / n, (tp + ts) / n);
}
