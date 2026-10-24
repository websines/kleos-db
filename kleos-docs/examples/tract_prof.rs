//! Per-op time of a recognizer run on tract (profiling aid).
use std::collections::HashMap;
use std::time::Instant;
use tract_onnx::prelude::*;
fn main() -> TractResult<()> {
    let a: Vec<String> = std::env::args().collect();
    let (path, b, h, w): (String, usize, usize, usize) = (a[1].clone(), a[2].parse()?, a[3].parse()?, a[4].parse()?);
    let model = tract_onnx::onnx().model_for_path(&path)?.into_optimized()?.into_runnable()?;
    let x = Tensor::zero::<f32>(&[b, 1, h, w])?;
    let ws = tensor1(&vec![w as i64; b]);
    for rep in 0..2 {
        let mut times: HashMap<String, (f64, usize)> = HashMap::new();
        let mut state = tract_core::plan::SimpleState::new(&model)?;
        let t0 = Instant::now();
        state.run_plan_with_eval(tvec!(x.clone().into(), ws.clone().into()), |ctx, st, node, input| -> TractResult<TVec<TValue>> {
            let t = Instant::now();
            let r = tract_core::plan::eval(ctx, st, node, input);
            let e = times.entry(format!("{}", node.op.name())).or_insert((0.0, 0));
            e.0 += t.elapsed().as_secs_f64() * 1000.0;
            e.1 += 1;
            r
        })?;
        if rep == 1 {
            let mut v: Vec<_> = times.into_iter().collect();
            v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            println!("total {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
            for (k, (t, n)) in v.iter().take(12) {
                println!("{t:8.1} ms  {n:3}x  {k}");
            }
        }
    }
    Ok(())
}
