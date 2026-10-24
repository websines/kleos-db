//! Chunking, modelled on Chonkie (chonkie-inc/chonkie, MIT; see NOTICE.md):
//! the token, sentence, recursive and semantic strategies on plain text, with
//! Chonkie's split and merge rules so the plain-text results match it, and a
//! document chunker that applies a strategy over layout blocks (tables kept
//! whole, headings attached to what follows, provenance per chunk).

use serde::{Deserialize, Serialize};

/// Token spans of a text: byte ranges, in order.
pub trait Tokenizer: Send + Sync {
    fn spans(&self, text: &str) -> Vec<(usize, usize)>;
    fn count(&self, text: &str) -> usize {
        self.spans(text).len()
    }
}

/// One token per character (Chonkie's default "character" tokenizer).
pub struct CharTokenizer;

impl Tokenizer for CharTokenizer {
    fn spans(&self, text: &str) -> Vec<(usize, usize)> {
        text.char_indices().map(|(i, c)| (i, i + c.len_utf8())).collect()
    }
    fn count(&self, text: &str) -> usize {
        text.chars().count()
    }
}

/// Tokens are the pieces of `text.split(' ')` (Chonkie's "word" tokenizer).
pub struct WordTokenizer;

impl Tokenizer for WordTokenizer {
    fn spans(&self, text: &str) -> Vec<(usize, usize)> {
        let mut out = vec![];
        let mut start = 0;
        for (i, b) in text.bytes().enumerate() {
            if b == b' ' {
                out.push((start, i));
                start = i + 1;
            }
        }
        out.push((start, text.len()));
        out
    }
    fn count(&self, text: &str) -> usize {
        text.bytes().filter(|&b| b == b' ').count() + 1
    }
}

/// About four characters per token (when no model tokenizer is at hand).
pub struct ApproxTokenizer;

impl Tokenizer for ApproxTokenizer {
    fn spans(&self, text: &str) -> Vec<(usize, usize)> {
        let idx: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        idx.chunks(4)
            .enumerate()
            .map(|(k, c)| (c[0], idx.get((k + 1) * 4).copied().unwrap_or(text.len())))
            .collect()
    }
    fn count(&self, text: &str) -> usize {
        text.chars().count().div_ceil(4)
    }
}

/// A chunk of plain text: its text, byte offsets and token count.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextChunk {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delim {
    Prev,
    Next,
}

/// Chonkie's `split_text_by_delimiters` (chonkie-core `split_offsets` /
/// `split_pattern_offsets`): cut after (`Prev`) or before (`Next`) every
/// delimiter, then merge: a piece joins the one before it while either is
/// shorter than `min_chars` bytes.  Empty pieces are dropped.
pub fn split_by_delimiters(text: &str, delims: &[&str], inc: Delim, min_chars: usize) -> Vec<String> {
    let b = text.as_bytes();
    let mut cuts: Vec<usize> = vec![];
    let mut i = 0;
    while i < b.len() {
        let mut hit = None;
        for d in delims {
            let db = d.as_bytes();
            if !db.is_empty() && b[i..].starts_with(db) {
                if hit.map(|h: usize| db.len() > h).unwrap_or(true) {
                    hit = Some(db.len());
                }
            }
        }
        if let Some(l) = hit {
            cuts.push(if inc == Delim::Prev { i + l } else { i });
            i += l;
        } else {
            i += 1;
        }
    }
    let mut pieces: Vec<(usize, usize)> = vec![];
    let mut s = 0;
    for c in cuts {
        if c > s {
            pieces.push((s, c));
            s = c;
        }
    }
    if s < b.len() {
        pieces.push((s, b.len()));
    }
    let mut out: Vec<(usize, usize)> = vec![];
    for p in pieces {
        if let Some(last) = out.last_mut() {
            if last.1 - last.0 < min_chars || p.1 - p.0 < min_chars {
                last.1 = p.1;
                continue;
            }
        }
        out.push(p);
    }
    out.into_iter().map(|(a, z)| text[a..z].to_string()).filter(|s| !s.is_empty()).collect()
}

/// chonkie-core `find_merge_indices`: greedy end indices of chunks whose
/// token counts sum to at most `chunk_size` (an oversized item is alone).
pub fn merge_indices(counts: &[usize], chunk_size: usize) -> Vec<usize> {
    let mut out = vec![];
    let mut acc = 0;
    for (i, &c) in counts.iter().enumerate() {
        if acc > 0 && acc + c > chunk_size {
            out.push(i);
            acc = 0;
        }
        acc += c;
    }
    if !counts.is_empty() {
        out.push(counts.len());
    }
    out
}

/// Fixed token windows with overlap (Chonkie's TokenChunker).
pub fn token_chunks(text: &str, tok: &dyn Tokenizer, size: usize, overlap: usize) -> Vec<TextChunk> {
    if text.trim().is_empty() {
        return vec![];
    }
    let spans = tok.spans(text);
    let step = size.saturating_sub(overlap).max(1);
    let mut out: Vec<TextChunk> = vec![];
    let mut cur = 0usize; // Chonkie's running index: end - overlap text length
    let mut start = 0;
    while start < spans.len() {
        let end = (start + size).min(spans.len());
        let (a, z) = (spans[start].0, spans[end - 1].1);
        let t = &text[a..z];
        let n = end - start;
        let tlen = t.chars().count();
        let ov = if overlap > 0 {
            let os = if n > overlap { end - overlap } else { start };
            text[spans[os].0..z].chars().count()
        } else {
            0
        };
        out.push(TextChunk { text: t.to_string(), start: cur, end: cur + tlen, tokens: n });
        cur = cur + tlen - ov;
        if end == spans.len() {
            break;
        }
        start += step;
    }
    out
}

pub const SENTENCE_DELIMS: &[&str] = &[". ", "! ", "? ", "\n"];

/// Sentences packed up to `size` tokens (Chonkie's SentenceChunker).
pub fn sentence_chunks(text: &str, tok: &dyn Tokenizer, size: usize, overlap: usize, min_sentences: usize, min_chars: usize) -> Vec<TextChunk> {
    if text.trim().is_empty() {
        return vec![];
    }
    let sents = split_by_delimiters(text, SENTENCE_DELIMS, Delim::Prev, min_chars);
    if sents.is_empty() {
        return vec![];
    }
    let mut pos = vec![];
    let mut p = 0;
    for s in &sents {
        pos.push(p);
        p += s.chars().count();
    }
    let counts: Vec<usize> = sents.iter().map(|s| tok.count(s)).collect();
    let mut out = vec![];
    let mut i = 0;
    while i < sents.len() {
        let idx = merge_indices(&counts[i..], size);
        let mut split = idx.first().map(|&k| i + k).unwrap_or(sents.len());
        if split - i < min_sentences {
            split = if i + min_sentences <= sents.len() { i + min_sentences } else { sents.len() };
        }
        let t: String = sents[i..split].concat();
        let n = tok.count(&t);
        out.push(TextChunk { start: pos[i], end: pos[split - 1] + sents[split - 1].chars().count(), text: t, tokens: n });
        if overlap > 0 && split < sents.len() {
            let mut ot = 0;
            let mut oi = split - 1;
            while oi > i && ot < overlap {
                let nt = ot + counts[oi] + 1;
                if nt > overlap {
                    break;
                }
                ot = nt;
                oi -= 1;
            }
            i = oi + 1;
        } else {
            i = split;
        }
    }
    out
}

/// One level of the recursive chunker.
#[derive(Clone, Debug)]
pub enum Level {
    Delims(Vec<&'static str>, Delim),
    Whitespace,
    Tokens,
}

/// Chonkie's default recursive rules: paragraphs, sentences, pauses, words, tokens.
pub fn default_levels() -> Vec<Level> {
    vec![
        Level::Delims(vec!["\n\n", "\r\n", "\n", "\r"], Delim::Prev),
        Level::Delims(vec![". ", "! ", "? "], Delim::Prev),
        Level::Delims(vec!["{", "}", "\"", "[", "]", "<", ">", "(", ")", ":", ";", ",", "—", "|", "~", "-", "...", "`", "'"], Delim::Prev),
        Level::Whitespace,
        Level::Tokens,
    ]
}

/// Chonkie's RecursiveChunker on plain text.
pub fn recursive_chunks(text: &str, tok: &dyn Tokenizer, size: usize, min_chars: usize, levels: &[Level]) -> Vec<TextChunk> {
    fn rec(text: &str, tok: &dyn Tokenizer, size: usize, min_chars: usize, levels: &[Level], level: usize, start: usize, out: &mut Vec<TextChunk>) {
        if text.is_empty() {
            return;
        }
        let mk = |t: &str, n: usize, s: usize| TextChunk { text: t.to_string(), start: s, end: s + t.chars().count(), tokens: n };
        if level >= levels.len() {
            out.push(mk(text, tok.count(text), start));
            return;
        }
        let splits: Vec<String> = match &levels[level] {
            Level::Delims(d, inc) => split_by_delimiters(text, d, *inc, min_chars),
            Level::Whitespace => split_by_delimiters(text, &[" "], Delim::Prev, min_chars),
            Level::Tokens => {
                let sp = tok.spans(text);
                sp.chunks(size).map(|c| text[c[0].0..c[c.len() - 1].1].to_string()).collect()
            }
        };
        let counts: Vec<usize> = splits.iter().map(|s| tok.count(s)).collect();
        let (merged, mc): (Vec<String>, Vec<usize>) = if matches!(levels[level], Level::Tokens) || counts.iter().all(|&c| c > size) {
            (splits, counts)
        } else {
            let idx = merge_indices(&counts, size);
            let mut m = vec![];
            let mut c = vec![];
            let mut s = 0;
            for e in idx {
                m.push(splits[s..e].concat());
                c.push(counts[s..e].iter().sum());
                s = e;
            }
            (m, c)
        };
        let mut off = start;
        for (sp, n) in merged.iter().zip(mc) {
            if n > size {
                rec(sp, tok, size, min_chars, levels, level + 1, off, out);
            } else {
                out.push(mk(sp, n, off));
            }
            off += sp.chars().count();
        }
    }
    let mut out = vec![];
    rec(text, tok, size, min_chars, levels, 0, 0, &mut out);
    out
}

/// Semantic chunking on plain text: sentences, a chunk ends where the cosine
/// similarity of neighbouring sentence windows falls below `threshold` (or
/// at `size` tokens).  `embed` returns one vector per text.
pub fn semantic_chunks(text: &str, tok: &dyn Tokenizer, size: usize, threshold: f32, window: usize, embed: &dyn Fn(&[String]) -> Vec<Vec<f32>>) -> Vec<TextChunk> {
    let sents = split_by_delimiters(text, SENTENCE_DELIMS, Delim::Prev, 12);
    if sents.is_empty() {
        return vec![];
    }
    let w = window.max(1);
    let wins: Vec<String> = (0..sents.len()).map(|i| sents[i.saturating_sub(w - 1)..=i].concat()).collect();
    let vecs = embed(&wins);
    let cos = |a: &[f32], b: &[f32]| {
        let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
        for (x, y) in a.iter().zip(b) {
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        d / (na.sqrt() * nb.sqrt()).max(1e-9)
    };
    let counts: Vec<usize> = sents.iter().map(|s| tok.count(s)).collect();
    let mut out = vec![];
    let (mut s0, mut acc, mut pos, mut start) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..sents.len() {
        let brk = i > s0 && (acc + counts[i] > size || (i < vecs.len() && cos(&vecs[i - 1], &vecs[i]) < threshold));
        if brk {
            let t = sents[s0..i].concat();
            let n = tok.count(&t);
            let l = t.chars().count();
            out.push(TextChunk { text: t, start, end: start + l, tokens: n });
            start += l;
            s0 = i;
            acc = 0;
        }
        acc += counts[i];
        pos += sents[i].chars().count();
    }
    let t = sents[s0..].concat();
    let n = tok.count(&t);
    out.push(TextChunk { end: pos, text: t, start, tokens: n });
    out
}

// ---------------------------------------------------------------- documents

/// A block of a document (a layout block of a page, or a Markdown block).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DocBlock {
    /// Text, Section-Header, Table, List-Group, Caption, Footnote, Picture, …
    pub label: String,
    /// 0-based page, when paged.
    pub page: Option<usize>,
    /// Box on its page in pixels (x0, y0, x1, y1).
    pub bbox: Option<[f32; 4]>,
    pub text: String,
    /// Heading level for headings (1 = top), from Markdown `#`s; OCR section headers are 2.
    pub level: Option<usize>,
}

/// A chunk of a document with its provenance.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Chunk {
    pub text: String,
    /// Byte offsets into the document Markdown (`doc_markdown` of the blocks).
    pub start: usize,
    pub end: usize,
    pub tokens: usize,
    pub page: Option<usize>,
    pub page_end: Option<usize>,
    /// (page, box) of each block it covers.
    pub boxes: Vec<(Option<usize>, [f32; 4])>,
    pub heading: Vec<String>,
    /// Indices of the blocks it covers.
    pub blocks: Vec<usize>,
    /// The dominant label (Table for a table chunk, else Text).
    pub label: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Strategy {
    Token,
    Sentence,
    Recursive,
    Semantic,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkOpts {
    pub strategy: Strategy,
    pub chunk_size: usize,
    pub overlap: usize,
    pub min_chars: usize,
    /// Semantic: similarity under which a chunk ends.
    pub threshold: f32,
}

impl Default for ChunkOpts {
    fn default() -> Self {
        ChunkOpts { strategy: Strategy::Recursive, chunk_size: 384, overlap: 0, min_chars: 24, threshold: 0.5 }
    }
}

/// The document Markdown the offsets refer to: block texts joined by blank lines.
pub fn doc_markdown(blocks: &[DocBlock]) -> (String, Vec<(usize, usize)>) {
    let mut s = String::new();
    let mut spans = vec![];
    for (i, b) in blocks.iter().enumerate() {
        if i > 0 {
            s.push_str("\n\n");
        }
        let a = s.len();
        s.push_str(&b.text);
        spans.push((a, s.len()));
    }
    (s, spans)
}

/// Markdown text to blocks: headings (`#`), fenced code, pipe tables and
/// paragraphs (blank-line separated).
pub fn markdown_blocks(md: &str) -> Vec<DocBlock> {
    let mut out = vec![];
    let mut para: Vec<&str> = vec![];
    let mut in_code = false;
    let mut code: Vec<&str> = vec![];
    let mut table: Vec<&str> = vec![];
    let flush = |para: &mut Vec<&str>, out: &mut Vec<DocBlock>, label: &str| {
        if !para.is_empty() {
            let t = para.join("\n");
            if !t.trim().is_empty() {
                out.push(DocBlock { label: label.into(), page: None, bbox: None, text: t, level: None });
            }
            para.clear();
        }
    };
    for line in md.lines() {
        if in_code {
            code.push(line);
            if line.trim_start().starts_with("```") {
                in_code = false;
                flush(&mut code, &mut out, "Code");
            }
            continue;
        }
        let tl = line.trim_start();
        if tl.starts_with("```") {
            flush(&mut para, &mut out, "Text");
            flush(&mut table, &mut out, "Table");
            in_code = true;
            code.push(line);
            continue;
        }
        if tl.starts_with('|') {
            flush(&mut para, &mut out, "Text");
            table.push(line);
            continue;
        } else if !table.is_empty() {
            flush(&mut table, &mut out, "Table");
        }
        if let Some(h) = tl.strip_prefix('#') {
            let lvl = 1 + h.chars().take_while(|&c| c == '#').count();
            let rest = h.trim_start_matches('#');
            if rest.starts_with(' ') || rest.is_empty() {
                flush(&mut para, &mut out, "Text");
                out.push(DocBlock { label: "Section-Header".into(), page: None, bbox: None, text: line.trim().to_string(), level: Some(lvl) });
                continue;
            }
        }
        if tl.is_empty() {
            flush(&mut para, &mut out, "Text");
        } else {
            para.push(line);
        }
    }
    flush(&mut table, &mut out, "Table");
    flush(&mut code, &mut out, "Code");
    flush(&mut para, &mut out, "Text");
    out
}

fn heading_text(b: &DocBlock) -> String {
    b.text.trim_start_matches('#').trim().to_string()
}

/// Chunk a document's blocks.  Headings never end a chunk and set the
/// heading path; tables and code blocks are never split; page headers,
/// footers and pictures are left out.
pub fn chunk_blocks(blocks: &[DocBlock], tok: &dyn Tokenizer, o: &ChunkOpts, embed: Option<&dyn Fn(&[String]) -> Vec<Vec<f32>>>) -> Vec<Chunk> {
    let (_, spans) = doc_markdown(blocks);
    // units: (block index, text, tokens, atomic, start offset)
    struct Unit {
        b: usize,
        text: String,
        tokens: usize,
        start: usize,
        end: usize,
        heading: bool,
    }
    let mut units: Vec<Unit> = vec![];
    let mut path_at: Vec<Vec<String>> = vec![];
    let mut path: Vec<(usize, String)> = vec![];
    for (i, b) in blocks.iter().enumerate() {
        if matches!(b.label.as_str(), "Page-Header" | "Page-Footer" | "Picture") || b.text.trim().is_empty() {
            path_at.push(path.iter().map(|p| p.1.clone()).collect());
            continue;
        }
        if b.label == "Section-Header" {
            let lvl = b.level.unwrap_or(2);
            path.retain(|p| p.0 < lvl);
            path.push((lvl, heading_text(b)));
            path_at.push(path.iter().map(|p| p.1.clone()).collect());
            units.push(Unit { b: i, text: b.text.clone(), tokens: tok.count(&b.text), start: spans[i].0, end: spans[i].1, heading: true });
            continue;
        }
        path_at.push(path.iter().map(|p| p.1.clone()).collect());
        let n = tok.count(&b.text);
        let atomic = matches!(b.label.as_str(), "Table" | "Code" | "Form" | "Table-Of-Contents");
        if atomic || n <= o.chunk_size {
            units.push(Unit { b: i, text: b.text.clone(), tokens: n, start: spans[i].0, end: spans[i].1, heading: false });
            continue;
        }
        // split a long block with the strategy, offsets relative to the block
        let parts = match o.strategy {
            Strategy::Token => token_chunks(&b.text, tok, o.chunk_size, o.overlap),
            Strategy::Sentence => sentence_chunks(&b.text, tok, o.chunk_size, o.overlap, 1, 12),
            Strategy::Semantic => match embed {
                Some(e) => semantic_chunks(&b.text, tok, o.chunk_size, o.threshold, 2, e),
                None => recursive_chunks(&b.text, tok, o.chunk_size, o.min_chars, &default_levels()),
            },
            Strategy::Recursive => recursive_chunks(&b.text, tok, o.chunk_size, o.min_chars, &default_levels()),
        };
        // char offsets inside the block -> byte offsets in the document
        let cidx: Vec<usize> = b.text.char_indices().map(|(k, _)| k).chain(std::iter::once(b.text.len())).collect();
        for p in parts {
            let s = spans[i].0 + cidx[p.start.min(cidx.len() - 1)];
            let e = spans[i].0 + cidx[p.end.min(cidx.len() - 1)];
            units.push(Unit { b: i, text: p.text, tokens: p.tokens, start: s, end: e, heading: false });
        }
    }
    let mut chunks: Vec<Chunk> = vec![];
    let mut cur: Vec<usize> = vec![];
    let mut acc = 0;
    let emit = |cur: &mut Vec<usize>, chunks: &mut Vec<Chunk>| {
        if cur.is_empty() {
            return;
        }
        // a chunk of headings only joins the next one
        if cur.iter().all(|&u| units[u].heading) {
            return;
        }
        let us: Vec<&Unit> = cur.iter().map(|&u| &units[u]).collect();
        let text = us.iter().map(|u| u.text.trim_end()).collect::<Vec<_>>().join("\n\n");
        let mut bl: Vec<usize> = us.iter().map(|u| u.b).collect();
        bl.dedup();
        let pages: Vec<usize> = bl.iter().filter_map(|&b| blocks[b].page).collect();
        let boxes = bl.iter().filter_map(|&b| blocks[b].bbox.map(|x| (blocks[b].page, x))).collect();
        let first_body = us.iter().find(|u| !u.heading).map(|u| u.b).unwrap_or(bl[0]);
        let label = if bl.iter().any(|&b| blocks[b].label == "Table") { "Table" } else { "Text" };
        chunks.push(Chunk {
            tokens: us.iter().map(|u| u.tokens).sum(),
            start: us[0].start,
            end: us[us.len() - 1].end,
            page: pages.iter().min().copied(),
            page_end: pages.iter().max().copied(),
            boxes,
            heading: path_at[first_body].clone(),
            blocks: bl,
            label: label.into(),
            text,
        });
        cur.clear();
    };
    for (ui, u) in units.iter().enumerate() {
        let atomic_big = u.tokens > o.chunk_size;
        let starts_section = u.heading && cur.iter().any(|&k| !units[k].heading);
        if !cur.is_empty() && (starts_section || atomic_big || acc + u.tokens > o.chunk_size) {
            // headings at the end of the current chunk move to the next one
            let mut carry = vec![];
            while let Some(&last) = cur.last() {
                if units[last].heading {
                    carry.insert(0, cur.pop().unwrap());
                } else {
                    break;
                }
            }
            emit(&mut cur, &mut chunks);
            cur = carry;
            acc = cur.iter().map(|&k| units[k].tokens).sum();
        }
        cur.push(ui);
        acc += u.tokens;
        if atomic_big {
            emit(&mut cur, &mut chunks);
            acc = 0;
        }
    }
    emit(&mut cur, &mut chunks);
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delimiter_split_matches_chonkie_core() {
        let t = "Hi. Ok. This is long sentence. Yes! No? End";
        assert_eq!(split_by_delimiters(t, SENTENCE_DELIMS, Delim::Prev, 12), vec![t.to_string()]);
        assert_eq!(split_by_delimiters(t, SENTENCE_DELIMS, Delim::Prev, 1), vec!["Hi. ", "Ok. ", "This is long sentence. ", "Yes! ", "No? ", "End"]);
        assert_eq!(split_by_delimiters(t, SENTENCE_DELIMS, Delim::Next, 1), vec!["Hi", ". Ok", ". This is long sentence", ". Yes", "! No", "? End"]);
        assert_eq!(split_by_delimiters("ab\n\ncd\nef", &["\n"], Delim::Prev, 3), vec!["ab\n\n", "cd\nef"]);
        assert_eq!(split_by_delimiters("x. y. zzzzzzzzzzzzzz. w", &[". "], Delim::Prev, 5), vec!["x. y. ", "zzzzzzzzzzzzzz. w"]);
    }

    #[test]
    fn semantic_breaks_at_topic_change() {
        let t = "Cats purr softly at night. Cats like warm laps a lot. Rockets need fuel to launch. Rockets burn oxygen fast.";
        let embed = |xs: &[String]| xs.iter().map(|x| if x.ends_with("lot. ") || x.contains("purr") && !x.contains("Rocket") { vec![1.0, 0.0] } else if x.contains("Cats") && x.contains("Rocket") { vec![0.7, 0.7] } else { vec![0.0, 1.0] }).collect::<Vec<_>>();
        let c = semantic_chunks(t, &WordTokenizer, 100, 0.8, 1, &embed);
        assert_eq!(c.len(), 2, "{c:?}");
        assert!(c[0].text.contains("laps") && c[1].text.starts_with("Rockets"));
    }

    #[test]
    fn merge_indices_matches_chonkie_core() {
        assert_eq!(merge_indices(&[3, 4, 5, 6, 2, 8], 10), vec![2, 3, 5, 6]);
        assert_eq!(merge_indices(&[3, 4, 5], 100), vec![3]);
        assert_eq!(merge_indices(&[12, 1, 1], 10), vec![1, 3]);
        assert_eq!(merge_indices(&[1, 1, 1, 1, 1, 1, 1], 3), vec![3, 6, 7]);
    }

    #[test]
    fn tables_stay_whole_and_headings_attach() {
        let md = "# Report\n\nIntro text here.\n\n## Numbers\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nAfter the table.";
        let blocks = markdown_blocks(md);
        assert_eq!(blocks.iter().map(|b| b.label.as_str()).collect::<Vec<_>>(), vec!["Section-Header", "Text", "Section-Header", "Table", "Text"]);
        let o = ChunkOpts { chunk_size: 12, ..Default::default() };
        let c = chunk_blocks(&blocks, &WordTokenizer, &o, None);
        // the table is one chunk unit, and "## Numbers" travels with it
        let t = c.iter().find(|c| c.text.contains("| 1 | 2 |")).unwrap();
        assert!(t.text.starts_with("## Numbers") && t.text.contains("| 3 | 4 |"), "{t:?}");
        assert_eq!(t.heading, vec!["Report", "Numbers"]);
        assert!(c.iter().all(|c| !c.text.trim_end().ends_with("## Numbers")));
        // a table bigger than the chunk size is one oversized chunk, not split
        let big = format!("| h |\n|---|\n{}", (0..40).map(|i| format!("| row {i} |")).collect::<Vec<_>>().join("\n"));
        let cb = chunk_blocks(&markdown_blocks(&big), &WordTokenizer, &o, None);
        assert_eq!(cb.len(), 1);
        assert_eq!(cb[0].label, "Table");
        let (md2, _) = doc_markdown(&blocks);
        for ch in &c {
            assert!(md2[ch.start..ch.end].contains(ch.text.lines().last().unwrap().trim()));
        }
    }
}
