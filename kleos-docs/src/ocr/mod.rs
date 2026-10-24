//! OCR: a Rust port of the Indra OCR pipeline (datalab-to/indra; code
//! Apache-2.0, see NOTICE.md), running its ONNX graphs on tract.
//!
//! The model weights (`datalab-to/indra-ocr` on Hugging Face) are under a
//! modified OpenRAIL-M licence: free for research, personal use and startups
//! under $20M funding or revenue, and not for use competitive with Datalab's
//! API.  They are downloaded on first use, never bundled.

pub mod convnet;
pub mod cv;
pub mod fastnn;
pub mod lm;
pub mod models;
pub mod order;
pub mod pipeline;
pub mod prep;
pub mod text;
#[cfg(feature = "pdf")]
pub mod textlayer;
#[cfg(feature = "pdf")]
pub mod document;

pub use pipeline::{Line, Ocr, Page};

/// An OCR engine for page images: what a scanned page (or an image) goes
/// through. Indra ([`Ocr`]) is the default and the one shipped; another
/// engine plugs in here. PDF text-layer pages keep using Indra's line
/// detector to place the layer's characters, so only full-OCR pages and
/// images use the backend.
pub trait OcrBackend: Send + Sync {
    /// Read one page image (`turn`: allow orientation correction).
    fn read_page(&self, gray: &cv::Gray, turn: bool) -> Result<Page, String>;
    /// A short name for provenance (`indra`).
    fn name(&self) -> &str;
}

impl OcrBackend for Ocr {
    fn read_page(&self, gray: &cv::Gray, turn: bool) -> Result<Page, String> {
        self.read(gray, turn).map_err(|e| e.to_string())
    }
    fn name(&self) -> &str {
        "indra"
    }
}

/// The Hugging Face cache folder (`HF_HUB_CACHE`, `$HF_HOME/hub`, or
/// `~/.cache/huggingface/hub`).
pub fn hf_cache_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("HF_HUB_CACHE") {
        return d.into();
    }
    if let Ok(h) = std::env::var("HF_HOME") {
        return std::path::Path::new(&h).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::Path::new(&home).join(".cache").join("huggingface").join("hub")
}

/// The OCR model folder: `KLEOS_OCR_MODEL_DIR`, else the pinned revision of
/// `datalab-to/indra-ocr` in the Hugging Face cache, downloaded on first use
/// (feature `download`; `HF_HUB_OFFLINE=1` forbids the download).
pub fn model_dir() -> Result<std::path::PathBuf, String> {
    if let Ok(d) = std::env::var("KLEOS_OCR_MODEL_DIR") {
        return Ok(d.into());
    }
    let snap_root = hf_cache_dir().join("models--datalab-to--indra-ocr");
    // a cached snapshot of the pinned revision
    let refs = snap_root.join("refs").join(pipeline::HF_REVISION);
    if let Ok(hash) = std::fs::read_to_string(&refs) {
        let d = snap_root.join("snapshots").join(hash.trim());
        if pipeline::MODEL_FILES.iter().all(|f| d.join(f).exists()) {
            return Ok(d);
        }
    }
    download_models()
}

#[cfg(feature = "download")]
fn download_models() -> Result<std::path::PathBuf, String> {
    let off = std::env::var("HF_HUB_OFFLINE").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
    if off {
        return Err("the OCR models are not cached and HF_HUB_OFFLINE forbids the download".into());
    }
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Cache, Repo, RepoType};
    let mut b = ApiBuilder::from_cache(Cache::new(hf_cache_dir())).with_progress(false);
    if let Ok(t) = std::env::var("HF_TOKEN") {
        b = b.with_token(Some(t));
    }
    let api = b.build().map_err(|e| e.to_string())?;
    let repo = api.repo(Repo::with_revision(pipeline::HF_REPO.into(), RepoType::Model, pipeline::HF_REVISION.into()));
    let mut dir = None;
    for f in pipeline::MODEL_FILES {
        let p = repo.get(f).map_err(|e| format!("{}: {f}: {e}", pipeline::HF_REPO))?;
        dir = p.parent().map(|d| d.to_path_buf());
    }
    for f in pipeline::OPTIONAL_FILES {
        let _ = repo.get(f);
    }
    dir.ok_or_else(|| "no model folder".into())
}

#[cfg(not(feature = "download"))]
fn download_models() -> Result<std::path::PathBuf, String> {
    Err("the OCR models are not cached; set KLEOS_OCR_MODEL_DIR or build with the `download` feature".into())
}
