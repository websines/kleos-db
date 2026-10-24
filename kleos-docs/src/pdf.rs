//! PDF access through pdfium (BSD-3 / Apache-2.0), loaded at run time from a
//! shared library: rendering pages to grayscale and reading their text layer
//! with character boxes and angles (what `pypdfium2` gives the Python
//! package, through the same pdfium calls).
//!
//! The library is found at `KLEOS_PDFIUM_LIB` (a file), in `KLEOS_PDFIUM_DIR`,
//! in the kleos cache (`pdfium/<build>/lib`, where `ensure_pdfium` downloads
//! the pdfium-binaries build), or on the system library path.  pdfium is not
//! thread safe: every call runs under one process-wide lock.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use pdfium_render::prelude::*;

use crate::ocr::cv::Gray;

const BITMAP_GRAY: i32 = 1;
const RENDER_FLAGS: i32 = 8 | 1; // FPDF_GRAYSCALE | FPDF_ANNOT
const OBJ_TEXT: u32 = 1;
const OBJ_FORM: u32 = 5;
const OBJ_PATH: u32 = 2;
const OBJ_IMAGE: u32 = 3;
const OBJ_SHADING: u32 = 4;
const TEXTMODE_INVISIBLE: i32 = 3;

/// The pdfium-binaries build the port was checked against (the one pypdfium2 5.14 ships).
pub const PDFIUM_BUILD: u32 = 8076;

#[derive(Debug, thiserror::Error)]
pub enum PdfError {
    #[error("pdfium library not found ({0}); set KLEOS_PDFIUM_LIB or allow the download")]
    NoLibrary(String),
    #[error("cannot open the PDF: {0}")]
    Open(String),
    #[error("password-protected PDF")]
    Locked,
    #[error("page {0}: {1}")]
    Page(usize, String),
}

struct Lib(Box<dyn PdfiumLibraryBindings>);
// SAFETY: every use goes through LIB's mutex, so pdfium never runs on two threads at once.
unsafe impl Send for Lib {}

static LIB: OnceLock<Result<Mutex<Lib>, String>> = OnceLock::new();
/// The file instance 0 was loaded from (copies of it make more instances).
static BASE: OnceLock<PathBuf> = OnceLock::new();
/// Extra pdfium instances: each a separate copy of the library, so its
/// global state is its own and it renders in parallel with the others.
static EXTRA: OnceLock<Mutex<Vec<&'static Mutex<Lib>>>> = OnceLock::new();

/// The kleos cache folder for native helpers (next to the Hugging Face cache).
pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("KLEOS_CACHE") {
        return PathBuf::from(d);
    }
    if let Ok(h) = std::env::var("HF_HOME") {
        if let Some(p) = Path::new(&h).parent() {
            return p.join("kleos");
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("kleos")
}

fn lib_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libpdfium.dylib"
    } else if cfg!(target_os = "windows") {
        "pdfium.dll"
    } else {
        "libpdfium.so"
    }
}

/// Candidate library paths, most specific first.
pub fn library_candidates() -> Vec<PathBuf> {
    let mut v = vec![];
    if let Ok(f) = std::env::var("KLEOS_PDFIUM_LIB") {
        v.push(PathBuf::from(f));
    }
    if let Ok(d) = std::env::var("KLEOS_PDFIUM_DIR") {
        v.push(Path::new(&d).join(lib_name()));
    }
    v.push(cache_dir().join("pdfium").join(PDFIUM_BUILD.to_string()).join("lib").join(lib_name()));
    v
}

fn open_lib() -> Result<Mutex<Lib>, String> {
    let mut tried = vec![];
    for p in library_candidates() {
        if p.exists() {
            match Pdfium::bind_to_library(&p) {
                Ok(b) => {
                    unsafe { b.FPDF_InitLibrary() };
                    let _ = BASE.set(p.clone());
                    return Ok(Mutex::new(Lib(b)));
                }
                Err(e) => tried.push(format!("{}: {e:?}", p.display())),
            }
        } else {
            tried.push(format!("{}: missing", p.display()));
        }
    }
    match Pdfium::bind_to_system_library() {
        Ok(b) => {
            unsafe { b.FPDF_InitLibrary() };
            Ok(Mutex::new(Lib(b)))
        }
        Err(e) => {
            tried.push(format!("system: {e:?}"));
            Err(tried.join("; "))
        }
    }
}

fn lib() -> Result<MutexGuard<'static, Lib>, PdfError> {
    lib_on(0)
}

/// Instance `k` (0 is the first loaded; k > 0 are copies, made on demand;
/// without a library file to copy, every k is instance 0).
fn instance(k: usize) -> Result<&'static Mutex<Lib>, PdfError> {
    let base = match LIB.get_or_init(open_lib) {
        Ok(m) => m,
        Err(e) => return Err(PdfError::NoLibrary(e.clone())),
    };
    if k == 0 {
        return Ok(base);
    }
    let Some(src) = BASE.get() else { return Ok(base) };
    let mut extra = EXTRA.get_or_init(|| Mutex::new(vec![])).lock().unwrap_or_else(|e| e.into_inner());
    while extra.len() < k {
        let n = extra.len() + 1;
        let dir = src.parent().unwrap_or(Path::new(".")).join("instances");
        let _ = std::fs::create_dir_all(&dir);
        let copy = dir.join(format!("{n}-{}", lib_name()));
        if !copy.exists() {
            let tmp = dir.join(format!(".{n}-{}.tmp{}", lib_name(), std::process::id()));
            std::fs::copy(src, &tmp).map_err(|e| PdfError::NoLibrary(format!("{}: {e}", copy.display())))?;
            let _ = std::fs::rename(&tmp, &copy);
        }
        let b = Pdfium::bind_to_library(&copy).map_err(|e| PdfError::NoLibrary(format!("{}: {e:?}", copy.display())))?;
        unsafe { b.FPDF_InitLibrary() };
        extra.push(Box::leak(Box::new(Mutex::new(Lib(b)))));
    }
    Ok(extra[k - 1])
}

fn lib_on(k: usize) -> Result<MutexGuard<'static, Lib>, PdfError> {
    Ok(instance(k)?.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Whether the pdfium library can be loaded.
pub fn available() -> bool {
    lib().is_ok()
}

/// An open PDF.  Pages are opened per call.
pub struct Doc {
    doc: FPDF_DOCUMENT,
    _bytes: Option<std::sync::Arc<Vec<u8>>>,
    path: Option<PathBuf>,
    inst: usize,
    pub pages: usize,
}

// SAFETY: the handle is only used under the library lock.
unsafe impl Send for Doc {}
// SAFETY: as above; pages are opened under the lock.
unsafe impl Sync for Doc {}

impl Doc {
    pub fn open(path: &Path, password: Option<&str>) -> Result<Doc, PdfError> {
        Self::open_on(0, Some(path), None, password)
    }

    pub fn from_bytes(bytes: Vec<u8>, password: Option<&str>) -> Result<Doc, PdfError> {
        Self::open_on(0, None, Some(std::sync::Arc::new(bytes)), password)
    }

    /// The same document opened on pdfium instance `k` (a worker's own).
    pub fn reopen_on(&self, k: usize) -> Result<Doc, PdfError> {
        if k == self.inst {
            return Self::open_on(k, self.path.as_deref(), self._bytes.clone(), None);
        }
        Self::open_on(k, self.path.as_deref(), self._bytes.clone(), None)
    }

    fn open_on(k: usize, path: Option<&Path>, bytes: Option<std::sync::Arc<Vec<u8>>>, password: Option<&str>) -> Result<Doc, PdfError> {
        let l = lib_on(k)?;
        let doc = unsafe {
            match (&bytes, path) {
                (Some(b), _) => l.0.FPDF_LoadMemDocument64(b, password),
                (None, Some(p)) => l.0.FPDF_LoadDocument(&p.to_string_lossy(), password),
                (None, None) => return Err(PdfError::Open("no source".into())),
            }
        };
        let mut d = Self::finish(&l, doc, bytes)?;
        d.path = path.map(|p| p.to_path_buf());
        d.inst = k;
        Ok(d)
    }

    fn finish(l: &Lib, doc: FPDF_DOCUMENT, bytes: Option<std::sync::Arc<Vec<u8>>>) -> Result<Doc, PdfError> {
        if doc.is_null() {
            let e = unsafe { l.0.FPDF_GetLastError() };
            return Err(if e == 4 { PdfError::Locked } else { PdfError::Open(format!("pdfium error {e}")) });
        }
        let pages = unsafe { l.0.FPDF_GetPageCount(doc) }.max(0) as usize;
        Ok(Doc { doc, _bytes: bytes, path: None, inst: 0, pages })
    }

    /// Run `f` with the page open (and the library locked).
    pub fn with_page<R>(&self, index: usize, f: impl FnOnce(&PageRef) -> R) -> Result<R, PdfError> {
        let l = lib_on(self.inst)?;
        let page = unsafe { l.0.FPDF_LoadPage(self.doc, index as i32) };
        if page.is_null() {
            return Err(PdfError::Page(index, "cannot load".into()));
        }
        let r = f(&PageRef { l: &l.0, page });
        unsafe { l.0.FPDF_ClosePage(page) };
        Ok(r)
    }
}

impl Drop for Doc {
    fn drop(&mut self) {
        if let Ok(l) = lib_on(self.inst) {
            unsafe { l.0.FPDF_CloseDocument(self.doc) };
        }
    }
}

/// What a page draws besides text (forms searched 4 deep): the cheap signal
/// for the digital-page fast path.
#[derive(Debug, Default, Clone)]
pub struct Census {
    pub images: usize,
    /// summed image bounds, in square points
    pub image_area: f64,
    pub paths: usize,
    pub shadings: usize,
}

/// A character of the text layer: its UTF-16 unit, angle and box (page space: l, b, r, t).
pub struct RawChar {
    pub unit: u32,
    pub angle: f32,
    pub bx: Option<[f64; 4]>,
}

/// An open page (valid inside `Doc::with_page`).
pub struct PageRef<'a> {
    l: &'a Box<dyn PdfiumLibraryBindings>,
    page: FPDF_PAGE,
}

impl PageRef<'_> {
    /// (width, height) in points.
    pub fn size(&self) -> (f64, f64) {
        unsafe { (self.l.FPDF_GetPageWidthF(self.page) as f64, self.l.FPDF_GetPageHeightF(self.page) as f64) }
    }

    pub fn rotation(&self) -> i32 {
        unsafe { self.l.FPDFPage_GetRotation(self.page) * 90 }
    }

    pub fn set_rotation(&self, degrees: i32) {
        unsafe { self.l.FPDFPage_SetRotation(self.page, (degrees / 90).rem_euclid(4)) }
    }

    /// Grayscale render at `scale` pixels per point (white fill, annotations
    /// drawn), sized ceil(points x scale) as pypdfium2 does.
    pub fn render_gray(&self, scale: f64) -> Gray {
        let (w, h) = self.size();
        let (rot, _) = (self.rotation(), ());
        let (mut bw, mut bh) = ((w * scale).ceil() as i32, (h * scale).ceil() as i32);
        // FPDF_GetPageWidthF already reports the rotated frame
        let _ = rot;
        bw = bw.max(1);
        bh = bh.max(1);
        unsafe {
            let bmp = self.l.FPDFBitmap_CreateEx(bw, bh, BITMAP_GRAY, std::ptr::null_mut(), 0);
            self.l.FPDFBitmap_FillRect(bmp, 0, 0, bw, bh, 0xFFFF_FFFF);
            self.l.FPDF_RenderPageBitmap(bmp, self.page, 0, 0, bw, bh, 0, RENDER_FLAGS);
            let stride = self.l.FPDFBitmap_GetStride(bmp) as usize;
            let buf = self.l.FPDFBitmap_GetBuffer(bmp) as *const u8;
            let mut data = Vec::with_capacity((bw * bh) as usize);
            for y in 0..bh as usize {
                let row = std::slice::from_raw_parts(buf.add(y * stride), bw as usize);
                data.extend_from_slice(row);
            }
            self.l.FPDFBitmap_Destroy(bmp);
            Gray::from_raw(bw as usize, bh as usize, data)
        }
    }

    /// [2, 3] map from page space to the rendered frame (points, top-left origin).
    pub fn device_affine(&self) -> [[f64; 3]; 2] {
        let (w, h) = self.size();
        let s = 1000.0;
        let dev = |x: f64, y: f64| -> (f64, f64) {
            let (mut dx, mut dy) = (0i32, 0i32);
            unsafe {
                self.l.FPDF_PageToDevice(self.page, 0, 0, (w * s).round() as i32, (h * s).round() as i32, 0, x, y, &mut dx, &mut dy);
            }
            (dx as f64 / s, dy as f64 / s)
        };
        let o = dev(0.0, 0.0);
        let ex = dev(1.0, 0.0);
        let ey = dev(0.0, 1.0);
        [[ex.0 - o.0, ey.0 - o.0, o.0], [ex.1 - o.1, ey.1 - o.1, o.1]]
    }

    /// Every text-layer character in content order.
    pub fn chars(&self) -> Vec<RawChar> {
        unsafe {
            let tp = self.l.FPDFText_LoadPage(self.page);
            if tp.is_null() {
                return vec![];
            }
            let n = self.l.FPDFText_CountChars(tp).max(0);
            let mut out = Vec::with_capacity(n as usize);
            for i in 0..n {
                let unit = self.l.FPDFText_GetUnicode(tp, i);
                let angle = self.l.FPDFText_GetCharAngle(tp, i);
                let (mut l, mut r, mut b, mut t) = (0f64, 0f64, 0f64, 0f64);
                let ok = self.l.FPDFText_GetCharBox(tp, i, &mut l, &mut r, &mut b, &mut t) != 0;
                out.push(RawChar { unit, angle, bx: if ok { Some([l, b, r, t]) } else { None } });
            }
            self.l.FPDFText_ClosePage(tp);
            out
        }
    }

    /// Images, paths and shadings on the page (forms searched 4 deep).
    pub fn census(&self) -> Census {
        let mut c = Census::default();
        unsafe {
            let count = self.l.FPDFPage_CountObjects(self.page);
            let mut stack: Vec<(FPDF_PAGEOBJECT, usize)> = (0..count).map(|i| (self.l.FPDFPage_GetObject(self.page, i), 1)).collect();
            while let Some((o, depth)) = stack.pop() {
                if o.is_null() {
                    continue;
                }
                match self.l.FPDFPageObj_GetType(o) as u32 {
                    OBJ_IMAGE => {
                        c.images += 1;
                        let (mut l, mut b, mut r, mut t) = (0f32, 0f32, 0f32, 0f32);
                        if self.l.FPDFPageObj_GetBounds(o, &mut l, &mut b, &mut r, &mut t) != 0 {
                            c.image_area += ((r - l).abs() * (t - b).abs()) as f64;
                        }
                    }
                    OBJ_PATH => c.paths += 1,
                    OBJ_SHADING => c.shadings += 1,
                    OBJ_FORM if depth < 4 => {
                        let n = self.l.FPDFFormObj_CountObjects(o).max(0);
                        for k in 0..n {
                            stack.push((self.l.FPDFFormObj_GetObject(o, k as _), depth + 1));
                        }
                    }
                    _ => {}
                }
            }
        }
        c
    }

    /// Share of text objects drawn invisibly (an OCR layer), searching form XObjects 4 deep.
    pub fn invisible_text_frac(&self) -> f64 {
        let (mut n, mut inv) = (0usize, 0usize);
        unsafe {
            let count = self.l.FPDFPage_CountObjects(self.page);
            let mut stack: Vec<(FPDF_PAGEOBJECT, usize)> = (0..count).map(|i| (self.l.FPDFPage_GetObject(self.page, i), 1)).collect();
            stack.reverse();
            while let Some((o, depth)) = stack.pop() {
                if o.is_null() {
                    continue;
                }
                let ty = self.l.FPDFPageObj_GetType(o) as u32;
                if ty == OBJ_TEXT {
                    n += 1;
                    if self.l.FPDFTextObj_GetTextRenderMode(o) as i32 == TEXTMODE_INVISIBLE {
                        inv += 1;
                    }
                } else if ty == OBJ_FORM && depth < 4 {
                    let c = self.l.FPDFFormObj_CountObjects(o).max(0);
                    for k in (0..c).rev() {
                        stack.push((self.l.FPDFFormObj_GetObject(o, k as _), depth + 1));
                    }
                }
            }
        }
        if n == 0 {
            0.0
        } else {
            inv as f64 / n as f64
        }
    }
}

/// Download the pdfium-binaries build for this platform into the kleos cache
/// (blocking; `KLEOS_OFFLINE=1` or `HF_HUB_OFFLINE=1` forbid it).  Returns the
/// library path.
#[cfg(feature = "download")]
pub fn ensure_pdfium() -> Result<PathBuf, PdfError> {
    for p in library_candidates() {
        if p.exists() {
            return Ok(p);
        }
    }
    let off = |k: &str| std::env::var(k).map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
    if off("KLEOS_OFFLINE") || off("HF_HUB_OFFLINE") {
        return Err(PdfError::NoLibrary("offline".into()));
    }
    let plat = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "mac-arm64",
        ("macos", "x86_64") => "mac-x64",
        ("linux", "x86_64") => "linux-x64",
        ("linux", "aarch64") => "linux-arm64",
        ("windows", "x86_64") => "win-x64",
        (o, a) => return Err(PdfError::NoLibrary(format!("no pdfium build for {o}/{a}"))),
    };
    let url = format!("https://github.com/bblanchon/pdfium-binaries/releases/download/chromium%2F{PDFIUM_BUILD}/pdfium-{plat}.tgz");
    let dest = cache_dir().join("pdfium").join(PDFIUM_BUILD.to_string());
    std::fs::create_dir_all(&dest).map_err(|e| PdfError::NoLibrary(e.to_string()))?;
    let tgz = dest.join("pdfium.tgz");
    let resp = ureq::get(&url).call().map_err(|e| PdfError::NoLibrary(format!("{url}: {e}")))?;
    let mut f = std::fs::File::create(&tgz).map_err(|e| PdfError::NoLibrary(e.to_string()))?;
    std::io::copy(&mut resp.into_reader(), &mut f).map_err(|e| PdfError::NoLibrary(e.to_string()))?;
    let st = std::process::Command::new("tar").arg("-xzf").arg(&tgz).arg("-C").arg(&dest).status().map_err(|e| PdfError::NoLibrary(e.to_string()))?;
    let _ = std::fs::remove_file(&tgz);
    if !st.success() {
        return Err(PdfError::NoLibrary("could not unpack pdfium".into()));
    }
    let p = dest.join("lib").join(lib_name());
    if p.exists() {
        Ok(p)
    } else {
        Err(PdfError::NoLibrary(format!("{} not in the archive", p.display())))
    }
}
