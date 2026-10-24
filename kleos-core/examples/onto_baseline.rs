//! The similarity baseline of the personal pack's prototypes under an
//! embedder (docs/specs/ontology.md, "Calibration").
//!
//! ```text
//! cargo +1.99.0 run --release --features infer-metal --example onto_baseline -- <model> [device]
//! ```
use kleos::ontology::{claim_anchors, proto_baseline, proto_neighbour, prototypes, Active, OntologyDoc, PERSONAL};
use kleos::DbInstance;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).cloned().unwrap_or_else(|| "embeddinggemma".into());
    let device = a.get(2).cloned().unwrap_or_else(|| "metal".into());
    let db = DbInstance::new("mem", "", "").unwrap();
    db.run_default(&format!("::embedder enable onto_b {{model: '{model}', device: '{device}'}}")).unwrap();
    let d: OntologyDoc = serde_json::from_str(PERSONAL).unwrap();
    let act = Active::merge(&[d], "cardinality", "decide");
    let ps = prototypes(&act, "onto_b").unwrap();
    println!("{model}: {} prototypes, baseline {:.4}, nearest-neighbour {:.4}", ps.len(), proto_baseline(&ps), proto_neighbour(&ps));
    println!("  claim anchors (mean, mean best): {:?}", claim_anchors(&ps));
}
