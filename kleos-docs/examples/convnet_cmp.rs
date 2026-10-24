//! The hand-written 2-D nets (detector, orientation) against tract on one
//! input: max |diff| per output and the time of each (warm, median of 5).
//!
//! `cargo run --release --example convnet_cmp -- <det.onnx|orient.onnx> <b> <h> <w>`
use std::time::Instant;

use kleos_docs::ocr::convnet::ConvNet;
use kleos_docs::ocr::models::ShapedModel;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = std::path::PathBuf::from(&a[1]);
    let shape: Vec<usize> = vec![a[2].parse().unwrap(), 1, a[3].parse().unwrap(), a[4].parse().unwrap()];
    let n: usize = shape.iter().product();
    let mut seed = 11u64;
    let x: Vec<f32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((seed >> 40) as f32) / ((1u64 << 24) as f32);
            if u < 0.12 { u * 3.0 } else { 0.95 }
        })
        .collect();
    let native = ConvNet::load(&path).expect("native load");
    let tract = ShapedModel::load(&path).expect("tract load");
    let mut tn = vec![];
    let mut tt = vec![];
    let (mut on, mut ot) = (vec![], vec![]);
    for _ in 0..6 {
        let t = Instant::now();
        on = native.run(&shape, x.clone()).unwrap();
        tn.push(t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        ot = tract.run(&shape, x.clone()).unwrap();
        tt.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    for v in [&mut tn, &mut tt] {
        v.remove(0);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    println!("native {:.1} ms, tract {:.1} ms", tn[2], tt[2]);
    for (k, ((sn, vn), (st, vt))) in on.iter().zip(ot.iter()).enumerate() {
        let d = vn.iter().zip(vt.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let m = vt.iter().map(|x| x.abs()).fold(0f32, f32::max);
        println!("output {k}: {sn:?} vs {st:?}, max |diff| {d:.2e} (max |value| {m:.2})");
    }
}
