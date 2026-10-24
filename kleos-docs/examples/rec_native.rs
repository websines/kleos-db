//! The hand-written recognizer against tract on one batch: agreement of every
//! output and the time of each (warm, median of 5).
//!
//! `cargo run --release --example rec_native -- <rec.onnx> <b> <h> <w>`
use std::time::Instant;

use kleos_docs::ocr::fastnn::RecNet;
use kleos_docs::ocr::models::{Input, Model, Output};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32) / ((1u64 << 24) as f32)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = std::path::PathBuf::from(&a[1]);
    let (b, h, w): (usize, usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
    let mut seed = 7u64;
    // ink-like input: mostly background with dark strokes, widths varying
    let x: Vec<f32> = (0..b * h * w).map(|_| if lcg(&mut seed) < 0.15 { lcg(&mut seed) * 0.4 } else { 0.9 + lcg(&mut seed) * 0.1 }).collect();
    let ws: Vec<i64> = (0..b).map(|i| (w - (i * 13) % (w / 3).max(1)) as i64).collect();
    let native = RecNet::load(&path).expect("native load");
    let tract = Model::load(&path).expect("tract load");
    let inputs = || vec![Input::F32(vec![b, 1, h, w], x.clone()), Input::I64(vec![b], ws.clone())];
    let time = |f: &mut dyn FnMut() -> Vec<Output>| {
        let mut ts = vec![];
        let mut out = vec![];
        for _ in 0..6 {
            let t = Instant::now();
            out = f();
            ts.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ts.remove(0);
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (ts[ts.len() / 2], out)
    };
    let (tn, on) = time(&mut || native.run(b, h, w, &x, &ws, true).unwrap());
    let (tt, ot) = time(&mut || tract.run(inputs()).unwrap());
    let (tg, _) = time(&mut || native.run(b, h, w, &x, &ws, false).unwrap());
    println!("native full {tn:.1} ms, native greedy {tg:.1} ms, tract full {tt:.1} ms");
    let names = ["ids", "conf", "lengths", "blank", "topv", "topi"];
    for (k, name) in names.iter().enumerate() {
        let (sn, st) = (on[k].shape(), ot[k].shape());
        if sn != st {
            println!("{name}: shape {sn:?} vs {st:?}");
            continue;
        }
        match (&on[k], &ot[k]) {
            (Output::I64(_, a), _) => {
                let bv = ot[k].i64();
                let same = a.iter().zip(bv.iter()).filter(|(x, y)| x == y).count();
                println!("{name}: {same}/{} equal", a.len());
            }
            (Output::F32(_, a), _) => {
                let bv = ot[k].f32();
                let d = a.iter().zip(bv.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                println!("{name}: max |diff| {d:.2e}");
            }
        }
    }
}
