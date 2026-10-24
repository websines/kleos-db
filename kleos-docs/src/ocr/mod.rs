//! OCR: a Rust port of the Indra OCR pipeline (datalab-to/indra; code
//! Apache-2.0, see NOTICE.md), running its ONNX graphs on tract.
//!
//! The model weights (`datalab-to/indra-ocr` on Hugging Face) are under a
//! modified OpenRAIL-M licence: free for research, personal use and startups
//! under $20M funding or revenue, and not for use competitive with Datalab's
//! API.  They are downloaded on first use, never bundled.

pub mod cv;
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
