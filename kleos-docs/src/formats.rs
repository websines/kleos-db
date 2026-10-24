//! Document types: detection from bytes and name, and the text formats
//! turned into Markdown (HTML, DOCX, plain text).

use std::io::Read;

/// What a document is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Pdf,
    Image,
    Markdown,
    Html,
    Docx,
    Text,
}

impl Kind {
    pub fn mime(self) -> &'static str {
        match self {
            Kind::Pdf => "application/pdf",
            Kind::Image => "image/*",
            Kind::Markdown => "text/markdown",
            Kind::Html => "text/html",
            Kind::Docx => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Kind::Text => "text/plain",
        }
    }
}

/// Detect the kind from the bytes (magic numbers first), then the mime type
/// or the name's extension.
pub fn detect(bytes: &[u8], mime: Option<&str>, name: Option<&str>) -> Option<Kind> {
    let head = &bytes[..bytes.len().min(1024)];
    if head.windows(5).any(|w| w == b"%PDF-") {
        return Some(Kind::Pdf);
    }
    let img = head.starts_with(b"\x89PNG")
        || head.starts_with(b"\xFF\xD8\xFF")
        || head.starts_with(b"II*\0")
        || head.starts_with(b"MM\0*")
        || head.starts_with(b"BM")
        || (head.len() > 12 && &head[..4] == b"RIFF" && &head[8..12] == b"WEBP")
        || head.starts_with(b"GIF8");
    if img {
        return Some(Kind::Image);
    }
    let ext = name.and_then(|n| n.rsplit_once('.').map(|x| x.1.to_ascii_lowercase()));
    let m = mime.unwrap_or("").to_ascii_lowercase();
    if head.starts_with(b"PK\x03\x04") && (ext.as_deref() == Some("docx") || m.contains("wordprocessingml") || zip_has(bytes, "word/document.xml")) {
        return Some(Kind::Docx);
    }
    if m.contains("pdf") {
        return Some(Kind::Pdf);
    }
    if m.contains("html") || matches!(ext.as_deref(), Some("html" | "htm" | "xhtml")) {
        return Some(Kind::Html);
    }
    if m.contains("markdown") || matches!(ext.as_deref(), Some("md" | "markdown" | "mdx")) {
        return Some(Kind::Markdown);
    }
    let text_like = std::str::from_utf8(bytes).is_ok();
    if !text_like {
        return None;
    }
    let s = String::from_utf8_lossy(head).to_ascii_lowercase();
    if s.trim_start().starts_with("<!doctype html") || s.trim_start().starts_with("<html") {
        return Some(Kind::Html);
    }
    if matches!(ext.as_deref(), Some("txt" | "text" | "log" | "csv" | "rst")) || m.starts_with("text/") || ext.is_none() {
        return Some(Kind::Text);
    }
    Some(Kind::Text)
}

fn zip_has(bytes: &[u8], member: &str) -> bool {
    zip::ZipArchive::new(std::io::Cursor::new(bytes)).map(|z| z.file_names().any(|n| n == member)).unwrap_or(false)
}

fn decode_entities(s: &str) -> String {
    s.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// HTML to Markdown: headings, paragraphs, list items and table rows; scripts
/// and styles dropped; inline tags stripped.
pub fn html_to_markdown(html: &str) -> String {
    use regex::Regex;
    use std::sync::OnceLock;
    static DROP: OnceLock<Regex> = OnceLock::new();
    static TAG: OnceLock<Regex> = OnceLock::new();
    let drop = DROP.get_or_init(|| Regex::new(r"(?is)<(script|style|noscript|head|svg)\b.*?</(script|style|noscript|head|svg)\s*>|<!--.*?-->").unwrap());
    let tag = TAG.get_or_init(|| Regex::new(r"(?is)<\s*(/?)\s*([a-z0-9]+)[^>]*>").unwrap());
    let html = drop.replace_all(html, "");
    let mut out = String::new();
    let mut last = 0;
    for m in tag.captures_iter(&html) {
        let whole = m.get(0).unwrap();
        out.push_str(&html[last..whole.start()]);
        last = whole.end();
        let closing = &m[1] == "/";
        let name = m[2].to_ascii_lowercase();
        match (name.as_str(), closing) {
            ("h1" | "h2" | "h3" | "h4" | "h5" | "h6", false) => {
                let n: usize = name[1..].parse().unwrap_or(1);
                out.push_str("\n\n");
                out.push_str(&"#".repeat(n));
                out.push(' ');
            }
            ("h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "p" | "div" | "section" | "article" | "table" | "ul" | "ol" | "blockquote" | "pre", _) => out.push_str("\n\n"),
            ("li", false) => out.push_str("\n- "),
            ("tr", false) => out.push_str("\n| "),
            ("td" | "th", true) => out.push_str(" | "),
            ("br", _) => out.push('\n'),
            _ => {}
        }
    }
    out.push_str(&html[last..]);
    let text = decode_entities(&out);
    // tidy: collapse spaces inside lines, at most one blank line
    let mut res = String::new();
    let mut blank = 0;
    for line in text.lines() {
        let l = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if l.is_empty() {
            blank += 1;
            if blank == 1 && !res.is_empty() {
                res.push('\n');
            }
            continue;
        }
        blank = 0;
        res.push_str(&l);
        res.push('\n');
    }
    res.trim().to_string()
}

/// DOCX to Markdown: `word/document.xml` paragraphs (Heading styles become
/// `#` headings, list paragraphs `- `) and tables as pipe rows.
pub fn docx_to_markdown(bytes: &[u8]) -> Result<String, String> {
    let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let mut xml = String::new();
    z.by_name("word/document.xml").map_err(|e| e.to_string())?.read_to_string(&mut xml).map_err(|e| e.to_string())?;
    use regex::Regex;
    let para = Regex::new(r"(?s)<w:p[ >].*?</w:p>|<w:p/>").unwrap();
    let text_run = Regex::new(r"(?s)<w:t(?: [^>]*)?>(.*?)</w:t>|<w:tab/>|<w:br/>").unwrap();
    let style = Regex::new(r#"<w:pStyle w:val="([^"]+)""#).unwrap();
    let table = Regex::new(r"(?s)<w:tbl>.*?</w:tbl>").unwrap();
    let row = Regex::new(r"(?s)<w:tr[ >].*?</w:tr>").unwrap();
    let cell = Regex::new(r"(?s)<w:tc>.*?</w:tc>").unwrap();
    let runs = |s: &str| -> String {
        text_run
            .captures_iter(s)
            .map(|c| match c.get(1) {
                Some(t) => decode_entities(t.as_str()),
                None => if c[0].contains("tab") { "\t".into() } else { "\n".into() },
            })
            .collect()
    };
    // tables become placeholders, rendered in place
    let mut tables = vec![];
    let body = table.replace_all(&xml, |c: &regex::Captures| {
        let t = &c[0];
        let rows: Vec<String> = row
            .find_iter(t)
            .map(|r| {
                let cells: Vec<String> = cell.find_iter(r.as_str()).map(|c| runs(c.as_str()).replace('\n', " ").trim().to_string()).collect();
                format!("| {} |", cells.join(" | "))
            })
            .collect();
        tables.push(rows.join("\n"));
        format!("<w:p><w:t>\u{1}TABLE{}\u{1}</w:t></w:p>", tables.len() - 1)
    });
    let mut out = vec![];
    for p in para.find_iter(&body) {
        let p = p.as_str();
        let t = runs(p);
        let t = t.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(rest) = t.strip_prefix('\u{1}').and_then(|r| r.strip_prefix("TABLE")) {
            let k: usize = rest.trim_end_matches('\u{1}').parse().unwrap_or(0);
            out.push(tables.get(k).cloned().unwrap_or_default());
            continue;
        }
        let st = style.captures(p).map(|c| c[1].to_ascii_lowercase()).unwrap_or_default();
        if let Some(n) = st.strip_prefix("heading").and_then(|d| d.trim().parse::<usize>().ok()) {
            out.push(format!("{} {t}", "#".repeat(n.clamp(1, 6))));
        } else if st == "title" {
            out.push(format!("# {t}"));
        } else if st.contains("list") || p.contains("<w:numPr>") {
            out.push(format!("- {t}"));
        } else {
            out.push(t.to_string());
        }
    }
    Ok(out.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_by_magic_then_name() {
        assert_eq!(detect(b"%PDF-1.7 ...", None, Some("x.bin")), Some(Kind::Pdf));
        assert_eq!(detect(b"\x89PNG\r\n\x1a\n....", None, None), Some(Kind::Image));
        assert_eq!(detect(b"# Title\n\nbody", None, Some("notes.md")), Some(Kind::Markdown));
        assert_eq!(detect(b"<!DOCTYPE html><html><body>x</body></html>", None, None), Some(Kind::Html));
        assert_eq!(detect(b"plain words", None, None), Some(Kind::Text));
        assert_eq!(detect(&[0u8, 159, 146, 150], None, None), None);
    }

    #[test]
    fn html_headings_lists_tables() {
        let md = html_to_markdown("<html><head><title>t</title></head><body><h2>Fees &amp; dues</h2><p>Some <b>bold</b> text.</p><ul><li>one</li><li>two</li></ul><table><tr><td>a</td><td>b</td></tr></table><script>x()</script></body></html>");
        assert!(md.contains("## Fees & dues"), "{md}");
        assert!(md.contains("Some bold text."));
        assert!(md.contains("- one") && md.contains("- two"));
        assert!(md.contains("| a | b |"));
        assert!(!md.contains("x()"));
    }
}
