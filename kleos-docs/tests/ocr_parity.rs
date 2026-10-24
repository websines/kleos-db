//! Parity with the Python Indra package on a reference set (needs the models,
//! pdfium and a reference made by scripts/docs/indra_ref.py), ignored by default:
//!
//!     KLEOS_OCR_MODEL_DIR=... KLEOS_PDFIUM_LIB=... KLEOS_OCR_PARITY=SAMPLES_DIR:REF.json \
//!       cargo +1.99.0 test -p kleos-docs --release -- --ignored
use std::path::PathBuf;

use kleos_docs::ocr::document::Converter;
use kleos_docs::ocr::Ocr;

#[test]
#[ignore]
fn same_lines_as_python() {
    let spec = std::env::var("KLEOS_OCR_PARITY").expect("KLEOS_OCR_PARITY=SAMPLES_DIR:REF.json");
    let (dir, refp) = spec.split_once(':').unwrap();
    let models = PathBuf::from(std::env::var("KLEOS_OCR_MODEL_DIR").expect("KLEOS_OCR_MODEL_DIR"));
    let ocr = Ocr::load(&models, 4).unwrap();
    // Indra parity: the full (render + detector) path; the digital fast path
    // is checked against it separately (examples/fastpath_cmp.rs)
    let mut conv = Converter::new(&ocr);
    conv.fast_digital = false;
    let reference: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refp).unwrap()).unwrap();
    let mut pages = 0;
    for (file, r) in reference.as_object().unwrap() {
        let want: Vec<Vec<String>> = r["pages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["lines"].as_array().unwrap().iter().map(|l| l["text"].as_str().unwrap().to_string()).collect())
            .collect();
        let mut got: Vec<Vec<String>> = vec![];
        conv.convert_path(&PathBuf::from(dir).join(file), 0, |_, p| got.push(p.lines.iter().map(|l| l.text.clone()).collect())).unwrap();
        assert_eq!(got, want, "{file}");
        pages += got.len();
    }
    assert!(pages > 0);
}
