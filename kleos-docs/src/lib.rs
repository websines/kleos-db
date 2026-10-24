//! Documents for kleos: OCR (a port of Indra OCR on tract), PDF text layers
//! and rendering (pdfium), layout and Markdown, and chunking.

pub mod chunk;
pub mod formats;

#[cfg(feature = "ocr")]
pub mod ocr;
#[cfg(feature = "pdf")]
pub mod pdf;
