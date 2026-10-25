/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The skill library (docs/specs/skills.md): Agent Skills folders stored as
//! immutable, content-addressed versions, served by task, measured by
//! outcome, learned and optimized behind a verification gate.
//!
//! - A **version** is the verbatim `SKILL.md` plus every file, with per-file
//!   `sha256:<hex>` digests and a manifest digest. Versions are never edited.
//! - The **head** of `(scope, name)` (`kleos_skill_head`, keyed by valid
//!   time) says which version is live; activation, rollback, retirement and
//!   undo assert a new head, so "which version was live at T" is a query.
//! - Only the live version of an active skill is indexed (FTS over its
//!   description and body chunks, HNSW over their embeddings when an embedder
//!   is configured); retrieval and the learned router are in `skill_find.rs`,
//!   the scanner in `skill_scan.rs`, signatures in `skill_sign.rs`, the
//!   learning loop and optimizer in `skill_learn.rs`.
//!
//! kleos stores and serves scripts; it never executes them.

use std::collections::{BTreeMap, BTreeSet};

use base64::Engine;
use miette::{bail, miette, IntoDiagnostic, Result};
use serde_json::{json, Value as JsonValue};
use sha2::{Digest, Sha256};

use crate::data::value::DataValue;
use crate::runtime::db::{Db, NamedRows, ScriptMutability};
use crate::storage::Storage;
use crate::JsonData;

pub const SKILL_META: &str = "kleos_skill_meta";
pub const SKILL: &str = "kleos_skill";
pub const SKILL_VER: &str = "kleos_skill_ver";
pub const SKILL_VSTAT: &str = "kleos_skill_vstat";
pub const SKILL_HEAD: &str = "kleos_skill_head";
pub const SKILL_BLOB: &str = "kleos_skill_blob";
pub const SKILL_CHUNK: &str = "kleos_skill_chunk";
pub const SKILL_VEC: &str = "kleos_skill_vec";
pub const SKILL_OUTCOME: &str = "kleos_skill_outcome";
pub const SKILL_USE: &str = "kleos_skill_use";
pub const SKILL_TRACE: &str = "kleos_skill_trace";
pub const SKILL_EVAL: &str = "kleos_skill_eval";
pub const SKILL_LOG: &str = "kleos_skill_log";
pub const SKILL_PROMO: &str = "kleos_skill_promo";
pub const SKILL_KEY: &str = "kleos_skill_key";
pub const SKILL_SIG: &str = "kleos_skill_sig";
pub const SKILL_PUB: &str = "kleos_skill_pub";
pub const SKILL_ROUTE: &str = "kleos_skill_route";
pub const SKILL_RUN: &str = "kleos_skill_run";

/// Version of the skill relation set (`kleos_skill_meta.version`). Separate
/// from the memory schema version: the skill relations are created on first
/// use and never touch the memory relations.
pub const SKILL_SCHEMA_VERSION: i64 = 1;
/// The scope of a store without tenancy.
pub const DEFAULT_SCOPE: &str = "default";
/// Flow queue of the skill workflows (`skill.learn`, `skill.optimize`).
pub const SKILL_QUEUE: &str = "skills";
pub const LEARN_WORKFLOW: &str = "skill.learn";
pub const OPTIMIZE_WORKFLOW: &str = "skill.optimize";

/// SEP-2640 limits a conforming host must accept.
pub const MAX_FILES: usize = 512;
pub const MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;
/// Body chunks for the index, at most this many characters.
const CHUNK_CHARS: usize = 1000;
/// Reference files indexed per skill (first chunk each).
const INDEXED_REFERENCES: usize = 4;
/// Index chunks per skill, at most (a very long SKILL.md is indexed by its
/// description and its first sections).
const MAX_CHUNKS: usize = 24;

const FTS_FILTERS: &str = "[Lowercase, Stopwords('en'), Stemmer('english')]";

const DDL: [(&str, &str); 18] = [
    (SKILL_META, "{key: String => value: Json}"),
    (SKILL, "{scope: String, name: String => ver: String?, status: String, description: String, created: Int, updated: Int, actor: Json}"),
    (
        SKILL_VER,
        "{ver: String => scope: String, name: String, seq: Int, digest: String, frontmatter: Json, body: String, files: Json, \
         parent: String?, origin: String, edits: Json?, actor: Json, provenance: Json, scan: Json, created: Int}",
    ),
    (SKILL_VSTAT, "{ver: String => status: String, at: Int, reason: String?}"),
    (SKILL_HEAD, "{scope: String, name: String, vt: Validity => ver: String?, status: String, actor: Json, reason: String?}"),
    (SKILL_BLOB, "{digest: String => bytes: Bytes}"),
    (SKILL_CHUNK, "{scope: String, name: String, n: Int => ver: String, kind: String, text: String}"),
    (
        SKILL_OUTCOME,
        "{ver: String, oid: String => scope: String, name: String, at: Int, run: String?, success: Bool, latency_ms: Float?, \
         tokens: Int?, cost: Float?, model: String?, task: String?, source: String, route: String?, actor: Json, metrics: Json?}",
    ),
    (SKILL_USE, "{run: String, ver: String => scope: String, name: String, at: Int, route: String?}"),
    (
        SKILL_TRACE,
        "{tid: String => scope: String, task: String, transcript: String, success: Bool, output: String?, run: String?, \
         at: Int, actor: Json, metrics: Json?, skills: [String]}",
    ),
    (SKILL_EVAL, "{scope: String, name: String, case: String => task: String, expected: String?, check: Json?, split: String, source: String, at: Int}"),
    (SKILL_LOG, "{scope: String, name: String, at: Int, eid: String => op: String, ver: String?, from: String?, actor: Json, detail: Json?}"),
    (
        SKILL_PROMO,
        "{id: String => ver: String, scope: String, name: String, to: String, status: String, actor: Json, approver: Json?, \
         at: Int, decided: Int?, reason: String?, new_ver: String?}",
    ),
    (SKILL_KEY, "{kid: String => public: String, label: String, trusted: Bool, at: Int}"),
    (SKILL_SIG, "{ver: String, kid: String => signature: String, at: Int}"),
    (SKILL_PUB, "{name: String, ver: String => scope: String, digest: String, package: Json, at: Int}"),
    (SKILL_ROUTE, "{rid: String => at: Int, task: String, scopes: [String], shown: Json}"),
    (SKILL_RUN, "{run: String => kind: String, scope: String, name: String?, state: Json, at: Int, updated: Int}"),
];

// ------------------------------------------------------------------ values

pub(crate) fn now_us() -> i64 {
    crate::flow::now_us()
}

pub(crate) fn new_id(prefix: &str) -> String {
    crate::flow::new_id(prefix)
}

pub(crate) fn sv(v: &str) -> DataValue {
    DataValue::from(v)
}

pub(crate) fn osv(v: Option<&str>) -> DataValue {
    v.map(DataValue::from).unwrap_or(DataValue::Null)
}

pub(crate) fn jd(v: &JsonValue) -> DataValue {
    match v {
        JsonValue::Null => DataValue::Null,
        v => DataValue::Json(JsonData(v.clone())),
    }
}

/// A `Json` (not null) column value.
pub(crate) fn jdn(v: &JsonValue) -> DataValue {
    DataValue::Json(JsonData(v.clone()))
}

pub(crate) fn jv(v: &DataValue) -> JsonValue {
    match v {
        DataValue::Json(j) => j.0.clone(),
        DataValue::Bytes(b) => JsonValue::String(base64::engine::general_purpose::STANDARD.encode(b)),
        v => JsonValue::from(v),
    }
}

pub(crate) fn cs(v: &DataValue) -> Option<String> {
    v.get_str().map(|s| s.to_string())
}

pub(crate) fn str_list(v: &[String]) -> DataValue {
    DataValue::List(v.iter().map(|s| DataValue::from(s.as_str())).collect())
}

pub(crate) fn params(p: Vec<(&str, DataValue)>) -> BTreeMap<String, DataValue> {
    p.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Rows of a result as JSON objects keyed by header.
pub(crate) fn rows_json(r: &NamedRows) -> Vec<serde_json::Map<String, JsonValue>> {
    r.rows
        .iter()
        .map(|row| r.headers.iter().zip(row.iter()).map(|(h, v)| (h.clone(), jv(v))).collect())
        .collect()
}

pub(crate) fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

/// `sha256:<64 lowercase hex>` (SEP-2640 digest format).
pub fn digest_of(b: &[u8]) -> String {
    format!("sha256:{}", sha256_hex(b))
}

/// Tokens of a text, estimated as one per four characters (the context
/// block's own estimate).
pub fn est_tokens(s: &str) -> usize {
    s.chars().count().div_ceil(4)
}

/// The actor of a call: `{human, agent, run}`. Every field is null in a
/// store without tenancy; `wave7/tenancy` fills it from the principal context.
pub fn actor_of(v: Option<&JsonValue>) -> JsonValue {
    let g = |k: &str| v.and_then(|x| x.get(k)).filter(|x| x.is_string()).cloned().unwrap_or(JsonValue::Null);
    let mut a = json!({"human": g("human"), "agent": g("agent"), "run": g("run")});
    if let Some(chain) = v.and_then(|x| x.get("chain")).filter(|c| c.is_array()) {
        a["chain"] = chain.clone();
    }
    a
}

// ------------------------------------------------------------------ scopes

/// A scope string, normalised: empty is the default scope.
pub fn norm_scope(s: Option<&str>) -> Result<String> {
    let s = s.map(str::trim).filter(|x| !x.is_empty()).unwrap_or(DEFAULT_SCOPE);
    if s.len() > 200 || !s.chars().all(|c| c.is_ascii_alphanumeric() || "._:~@+-".contains(c)) {
        bail!("skill scope '{s}' has characters outside [A-Za-z0-9._:~@+-]");
    }
    Ok(s.to_string())
}

/// Kind of a scope, after the tenancy scope table: `run`, `agent_user`,
/// `user`, `agent`, `team`, `org`, else `workspace` (the workspace scope and
/// every legacy namespace).
pub fn scope_kind(s: &str) -> &'static str {
    match s.split(':').next().unwrap_or("") {
        "run" => "run",
        "au" => "agent_user",
        "sess" => "run",
        "user" => "user",
        "agent" => "agent",
        "team" => "team",
        "org" if !s.contains(':') => "org",
        _ => "workspace",
    }
}

/// Precedence of a scope, narrowest (0) to broadest (6).
pub fn scope_rank(s: &str) -> u8 {
    match scope_kind(s) {
        "run" => 0,
        "agent_user" => 1,
        "user" => 2,
        "agent" => 3,
        "team" => 4,
        "workspace" => 5,
        _ => 6,
    }
}

/// The `skill://` path prefix of a scope: none for the default scope, else
/// its `:`-separated parts as path segments.
pub fn scope_prefix(scope: &str) -> String {
    if scope == DEFAULT_SCOPE {
        return String::new();
    }
    scope.split(':').map(pct).collect::<Vec<_>>().join("/")
}

/// `skill://<scope prefix>/<name>/<path>`.
pub fn skill_uri(scope: &str, name: &str, path: &str) -> String {
    let p = scope_prefix(scope);
    if p.is_empty() {
        format!("skill://{name}/{path}")
    } else {
        format!("skill://{p}/{name}/{path}")
    }
}

pub(crate) fn pct(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'@' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

pub(crate) fn unpct(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Some(x) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(x);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The `(scope, name, rest)` splits a `skill://` URI could mean, broadest
/// prefix last: the caller keeps the first split naming a skill it has.
pub fn uri_splits(uri: &str) -> Vec<(String, String, String)> {
    let Some(rest) = uri.strip_prefix("skill://") else { return vec![] };
    let segs: Vec<&str> = rest.trim_end_matches('/').split('/').collect();
    let mut out = vec![];
    for i in 0..segs.len() {
        let name = unpct(segs[i]);
        if validate_name(&name).is_err() {
            continue;
        }
        let scope = if i == 0 { DEFAULT_SCOPE.to_string() } else { segs[..i].iter().map(|x| unpct(x)).collect::<Vec<_>>().join(":") };
        out.push((scope, name, segs[i + 1..].join("/")));
    }
    out
}

// ------------------------------------------------------------------ format

/// A parsed `SKILL.md`.
#[derive(Debug, Clone)]
pub struct SkillDoc {
    pub name: String,
    pub description: String,
    /// The YAML frontmatter rendered as JSON, verbatim (every field).
    pub frontmatter: JsonValue,
    /// The frontmatter's YAML text, between the `---` lines.
    pub fm_text: String,
    pub body: String,
}

/// Agent Skills name rules: 1-64 characters of lowercase letters, digits and
/// hyphens, no leading, trailing or doubled hyphen.
pub fn validate_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--");
    if !ok {
        bail!("skill name '{name}' must be 1-64 lowercase letters, digits and single hyphens (agentskills.io)");
    }
    Ok(())
}

/// Parse and validate a `SKILL.md` (agentskills.io rules).
pub fn parse_skill_md(text: &str) -> Result<SkillDoc> {
    let t = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = t.split_inclusive('\n');
    let first = lines.next().unwrap_or("");
    if first.trim_end() != "---" {
        bail!("SKILL.md must begin with YAML frontmatter between '---' lines");
    }
    let mut fm = String::new();
    let mut consumed = first.len();
    let mut closed = false;
    for l in lines {
        consumed += l.len();
        if l.trim_end() == "---" || l.trim_end() == "..." {
            closed = true;
            break;
        }
        fm.push_str(l);
    }
    if !closed {
        bail!("SKILL.md frontmatter is not closed by a '---' line");
    }
    let body = t[consumed..].to_string();
    let yaml: serde_yaml::Value = serde_yaml::from_str(&fm).map_err(|e| miette!("SKILL.md frontmatter is not valid YAML: {e}"))?;
    let frontmatter: JsonValue = serde_json::to_value(&yaml).map_err(|e| miette!("SKILL.md frontmatter cannot be rendered as JSON: {e}"))?;
    let Some(obj) = frontmatter.as_object() else { bail!("SKILL.md frontmatter must be a mapping") };
    let name = obj.get("name").and_then(|v| v.as_str()).ok_or_else(|| miette!("SKILL.md frontmatter needs 'name'"))?.to_string();
    validate_name(&name)?;
    let description = obj
        .get("description")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| miette!("SKILL.md frontmatter needs a non-empty 'description'"))?;
    if description.chars().count() > 1024 {
        bail!("skill '{name}': description is longer than 1024 characters");
    }
    if let Some(c) = obj.get("compatibility").and_then(|v| v.as_str()) {
        if c.chars().count() > 500 {
            bail!("skill '{name}': compatibility is longer than 500 characters");
        }
    }
    Ok(SkillDoc { name, description, frontmatter, fm_text: fm, body })
}

/// A `SKILL.md` with the same frontmatter text and a new body.
pub fn with_body(doc: &SkillDoc, body: &str) -> String {
    format!("---\n{}---\n{}", doc.fm_text, body)
}

/// One file of a version's manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct FileEntry {
    pub path: String,
    pub digest: String,
    pub size: usize,
}

/// Kind of a file by its place in the folder.
pub fn file_kind(path: &str) -> &'static str {
    if path == "SKILL.md" {
        "skill"
    } else if path.starts_with("scripts/") {
        "script"
    } else if path.starts_with("references/") {
        "reference"
    } else if path.starts_with("assets/") {
        "asset"
    } else {
        "other"
    }
}

/// A relative file path inside a skill: `/`-separated, no `..`, no absolute
/// or hidden-control parts.
pub fn check_path(p: &str) -> Result<()> {
    let bad = p.is_empty()
        || p.starts_with('/')
        || p.contains('\\')
        || p.split('/').any(|s| s.is_empty() || s == "." || s == "..")
        || p.chars().any(|c| c.is_control());
    if bad {
        bail!("skill file path '{p}' must be relative, '/'-separated, without '..'");
    }
    Ok(())
}

/// The manifest of a file set (sorted by path) and its digest: SHA-256 of the
/// canonical manifest JSON `{"files":[{"digest","path","size"},…]}`.
pub fn manifest(files: &BTreeMap<String, Vec<u8>>) -> (Vec<FileEntry>, String) {
    let entries: Vec<FileEntry> =
        files.iter().map(|(p, b)| FileEntry { path: p.clone(), digest: digest_of(b), size: b.len() }).collect();
    let d = digest_of(&manifest_bytes(&entries));
    (entries, d)
}

/// The canonical manifest bytes a version digest (and a signature) covers.
pub fn manifest_bytes(entries: &[FileEntry]) -> Vec<u8> {
    let mut e = entries.to_vec();
    e.sort_by(|a, b| a.path.cmp(&b.path));
    let files: Vec<String> = e
        .iter()
        .map(|f| {
            format!(
                "{{\"digest\":{},\"path\":{},\"size\":{}}}",
                JsonValue::String(f.digest.clone()),
                JsonValue::String(f.path.clone()),
                f.size
            )
        })
        .collect();
    format!("{{\"files\":[{}]}}", files.join(",")).into_bytes()
}

pub(crate) fn manifest_json(entries: &[FileEntry]) -> JsonValue {
    JsonValue::Array(
        entries
            .iter()
            .map(|f| {
                let kind = file_kind(&f.path);
                let mut o = json!({"path": f.path, "digest": f.digest, "size": f.size, "kind": kind});
                if kind == "script" {
                    // stored and served, never executed by kleos
                    o["exec"] = json!("requires-approval");
                }
                o
            })
            .collect(),
    )
}

pub(crate) fn entries_of(files: &JsonValue) -> Vec<FileEntry> {
    files
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|f| {
                    Some(FileEntry {
                        path: f.get("path")?.as_str()?.to_string(),
                        digest: f.get("digest")?.as_str()?.to_string(),
                        size: f.get("size")?.as_u64()? as usize,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Permissions a version declares: `metadata` keys `kleos.permissions`
/// (comma- or space-separated: `network`, `shell`, `fs-write`, `secrets`) and
/// the experimental `allowed-tools` (recorded, never honoured for served
/// skills).
pub fn permissions_of(frontmatter: &JsonValue) -> JsonValue {
    let declared: Vec<String> = frontmatter
        .get("metadata")
        .and_then(|m| m.get("kleos.permissions").or_else(|| m.get("permissions")))
        .and_then(|v| v.as_str())
        .map(|s| s.split(|c: char| c == ',' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.to_lowercase()).collect())
        .unwrap_or_default();
    json!({"declared": declared, "allowed_tools": frontmatter.get("allowed-tools").cloned().unwrap_or(JsonValue::Null)})
}

/// Index chunks of a version: its description, then the body cut at headings
/// and paragraphs into pieces of at most [`CHUNK_CHARS`] characters, then the
/// first piece of up to [`INDEXED_REFERENCES`] text references.
pub(crate) fn chunks_of(name: &str, description: &str, body: &str, refs: &[(String, String)]) -> Vec<(String, String)> {
    let mut out = vec![("desc".to_string(), format!("{name}: {description}"))];
    let mut heading = String::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, heading: &str, out: &mut Vec<(String, String)>| {
        let t = cur.trim();
        if !t.is_empty() {
            let h = if heading.is_empty() { String::new() } else { format!(" ({heading})") };
            out.push(("body".to_string(), format!("{name}{h}: {t}")));
        }
        cur.clear();
    };
    let mut in_code = false;
    for para in body.split("\n\n") {
        let p = para.trim();
        if p.is_empty() {
            continue;
        }
        if p.starts_with("```") || p.matches("```").count() % 2 == 1 {
            in_code = !in_code || p.matches("```").count() % 2 == 0;
        }
        if !in_code && p.starts_with('#') {
            flush(&mut cur, &heading, &mut out);
            let first = p.lines().next().unwrap_or("");
            heading = first.trim_start_matches('#').trim().to_string();
            let rest: String = p.lines().skip(1).collect::<Vec<_>>().join("\n");
            if !rest.trim().is_empty() {
                cur.push_str(rest.trim());
            }
            continue;
        }
        if !cur.is_empty() && cur.chars().count() + p.chars().count() > CHUNK_CHARS {
            flush(&mut cur, &heading, &mut out);
        }
        if p.chars().count() > CHUNK_CHARS {
            // one long paragraph: cut at sentence or line ends
            let mut piece = String::new();
            for s in p.split_inclusive(['.', '\n']) {
                if piece.chars().count() + s.chars().count() > CHUNK_CHARS && !piece.is_empty() {
                    cur.push_str(&piece);
                    flush(&mut cur, &heading, &mut out);
                    piece.clear();
                }
                piece.push_str(s);
            }
            cur.push_str(&piece);
        } else {
            if !cur.is_empty() {
                cur.push_str("\n\n");
            }
            cur.push_str(p);
        }
    }
    flush(&mut cur, &heading, &mut out);
    out.truncate(MAX_CHUNKS - refs.len().min(INDEXED_REFERENCES));
    for (path, text) in refs.iter().take(INDEXED_REFERENCES) {
        let t: String = text.chars().take(CHUNK_CHARS).collect();
        if !t.trim().is_empty() {
            out.push(("res".to_string(), format!("{name} ({path}): {}", t.trim())));
        }
    }
    out
}

// ------------------------------------------------------------------ folders

/// Read one skill folder: every regular file under it (hidden files and
/// directories skipped), as `relative path → bytes`.
pub fn read_skill_dir(dir: &std::path::Path) -> Result<BTreeMap<String, Vec<u8>>> {
    fn walk(base: &std::path::Path, dir: &std::path::Path, out: &mut BTreeMap<String, Vec<u8>>, total: &mut usize) -> Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(dir).into_diagnostic()?.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let ft = e.file_type().into_diagnostic()?;
            let p = e.path();
            if ft.is_symlink() {
                // a link could point anywhere on the host: never followed
                continue;
            }
            if ft.is_dir() {
                walk(base, &p, out, total)?;
            } else if ft.is_file() {
                let rel = p.strip_prefix(base).into_diagnostic()?.to_string_lossy().replace('\\', "/");
                let b = std::fs::read(&p).into_diagnostic()?;
                *total += b.len();
                if out.len() >= MAX_FILES || *total > MAX_TOTAL_BYTES {
                    bail!("skill folder {} is over the SEP-2640 limits ({MAX_FILES} files, 16 MiB)", base.display());
                }
                out.insert(rel, b);
            }
        }
        Ok(())
    }
    let mut out = BTreeMap::new();
    let mut total = 0;
    walk(dir, dir, &mut out, &mut total)?;
    if !out.contains_key("SKILL.md") {
        bail!("{} has no SKILL.md", dir.display());
    }
    Ok(out)
}

/// The skill folders under `path`: itself when it holds a `SKILL.md`; a
/// Claude Code plugin's skills (`.claude-plugin/plugin.json`, `skills`
/// field or `skills/`); else every child folder holding a `SKILL.md`.
pub fn skill_dirs(path: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    if path.join("SKILL.md").is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut roots = vec![path.to_path_buf()];
    let plugin = path.join(".claude-plugin").join("plugin.json");
    if plugin.is_file() {
        let v: JsonValue = serde_json::from_slice(&std::fs::read(&plugin).into_diagnostic()?).into_diagnostic()?;
        roots = match v.get("skills") {
            Some(JsonValue::String(s)) => vec![path.join(s)],
            Some(JsonValue::Array(a)) => a.iter().filter_map(|x| x.as_str()).map(|s| path.join(s)).collect(),
            _ => vec![path.join("skills")],
        };
    } else if path.join("skills").is_dir() {
        roots.push(path.join("skills"));
    }
    let mut out = vec![];
    for r in roots {
        if r.join("SKILL.md").is_file() {
            out.push(r);
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&r) else { continue };
        let mut ds: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join("SKILL.md").is_file()).collect();
        ds.sort();
        out.extend(ds);
    }
    out.dedup();
    if out.is_empty() {
        bail!("no SKILL.md found under {}", path.display());
    }
    Ok(out)
}

// ------------------------------------------------------------------ options

fn from_args<T: serde::de::DeserializeOwned>(op: &str, args: &JsonValue) -> Result<T> {
    let args = if args.is_null() { json!({}) } else { args.clone() };
    serde_json::from_value(args).map_err(|e| miette!("skill {op}: bad options: {e}"))
}

fn arg_str<'a>(a: &'a JsonValue, k: &str) -> Option<&'a str> {
    a.get(k).and_then(|v| v.as_str()).filter(|s| !s.is_empty())
}

/// Lifecycle settings (`::skill config {lifecycle: {…}}`).
#[derive(Debug, Clone, serde_derive::Serialize, serde_derive::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LifecycleOpts {
    /// Roll a new version back when it does worse than its predecessor.
    pub auto_rollback: bool,
    /// Outcomes a version needs before it is compared or retired.
    pub min_samples: i64,
    /// Posterior probability that the new version is worse, to roll back.
    pub rollback_confidence: f64,
    /// Retire active skills unused for this many days (0: never).
    pub retire_unused_days: i64,
    /// Retire a skill whose success rate is below this with 95% confidence.
    pub retire_below: f64,
}

impl Default for LifecycleOpts {
    fn default() -> Self {
        Self { auto_rollback: true, min_samples: 20, rollback_confidence: 0.95, retire_unused_days: 90, retire_below: 0.25 }
    }
}

/// Promotion policy (`::skill config {policy: {…}}`): promotion into these
/// scope kinds needs no approval when the version qualifies. Empty by default:
/// skills are instance knowledge, so every promotion is approved.
#[derive(Debug, Clone, Default, serde_derive::Serialize, serde_derive::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PolicyOpts {
    pub auto_promote_to: Vec<String>,
    pub auto_min_uses: i64,
    pub auto_min_success: f64,
    /// Refuse to install unsigned packages (default true).
    pub require_signature: Option<bool>,
}

// ------------------------------------------------------------------ engine

impl<'s, S: Storage<'s>> Db<S> {
    pub(crate) fn sk_r(&'s self, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
        self.run_script(script, params(p), ScriptMutability::Immutable)
    }

    /// A write, retried while it loses to concurrent writers.
    pub(crate) fn sk_w(&'s self, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
        let p = params(p);
        let mut attempt = 0;
        loop {
            match self.run_script(script, p.clone(), ScriptMutability::Mutable) {
                Ok(r) => return Ok(r),
                Err(e) if attempt < 10 && crate::memory::is_contention(&e) => {
                    std::thread::sleep(std::time::Duration::from_millis(1 << attempt.min(6)));
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    pub(crate) fn sk_has(&'s self, name: &str) -> bool {
        self.transact().map(|tx| tx.get_relation(name, false).is_ok()).unwrap_or(false)
    }

    /// Create the skill relations that do not exist yet (first use).
    pub(crate) fn skill_ensure(&'s self) -> Result<()> {
        if self.sk_has(SKILL_RUN) && self.sk_has(SKILL_META) {
            return Ok(());
        }
        let missing: Vec<&(&str, &str)> = {
            let tx = self.transact()?;
            DDL.iter().filter(|(n, _)| tx.get_relation(n, false).is_err()).collect()
        };
        for (name, cols) in missing {
            if let Err(e) = self.run_script(&format!(":create {name} {cols}"), Default::default(), ScriptMutability::Mutable) {
                if !self.sk_has(name) {
                    return Err(e);
                }
            }
            if *name == SKILL_CHUNK {
                let _ = self.run_script(
                    &format!("::fts create {SKILL_CHUNK}:fts {{extractor: text, tokenizer: Simple, filters: {FTS_FILTERS}}}"),
                    Default::default(),
                    ScriptMutability::Mutable,
                );
            }
        }
        if self.skill_meta("version")?.is_none() {
            self.set_skill_meta("version", &json!(SKILL_SCHEMA_VERSION))?;
        }
        Ok(())
    }

    pub(crate) fn skill_meta(&'s self, key: &str) -> Result<Option<JsonValue>> {
        let r = self.sk_r(&format!("?[value] := *{SKILL_META}{{key: $k, value}}"), vec![("k", sv(key))])?;
        Ok(r.rows.first().map(|x| jv(&x[0])))
    }

    pub(crate) fn set_skill_meta(&'s self, key: &str, v: &JsonValue) -> Result<()> {
        self.sk_w(&format!("?[key, value] <- [[$k, $v]] :put {SKILL_META} {{key => value}}"), vec![("k", sv(key)), ("v", jdn(v))])?;
        Ok(())
    }

    pub(crate) fn skill_lifecycle_opts(&'s self) -> LifecycleOpts {
        self.skill_meta("lifecycle").ok().flatten().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
    }

    pub(crate) fn skill_policy(&'s self) -> PolicyOpts {
        self.skill_meta("policy").ok().flatten().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
    }

    /// The embedder of the skill index: the configured one, else the memory
    /// store's, else none (FTS only).
    pub(crate) fn skill_embedder(&'s self) -> Option<(String, usize)> {
        let name = self.skill_meta("embedder").ok().flatten().and_then(|v| v.as_str().map(String::from));
        let dim = self.skill_meta("dim").ok().flatten().and_then(|v| v.as_u64()).map(|d| d as usize);
        match (name, dim) {
            (Some(n), Some(d)) if crate::embed::is_embedder_enabled(&n) => Some((n, d)),
            _ => None,
        }
    }

    /// Attach an embedder to the skill index: create the vector relation and
    /// its HNSW index, and embed every indexed chunk.
    fn skill_attach_embedder(&'s self, name: &str) -> Result<JsonValue> {
        let st = crate::embed::embedders()
            .into_iter()
            .find(|e| e.name == name && e.enabled)
            .ok_or_else(|| miette!("embedder '{name}' is not enabled; enable it with `::embedder enable {name}` first"))?;
        let old_dim = self.skill_meta("dim")?.and_then(|v| v.as_u64()).map(|d| d as usize);
        let old_model = self.skill_meta("embedder_model")?.and_then(|v| v.as_str().map(String::from));
        if self.sk_has(SKILL_VEC) && (old_dim != Some(st.dim) || old_model.as_deref() != Some(st.model.as_str())) {
            let _ = self.sk_w(&format!("::hnsw drop {SKILL_VEC}:hnsw"), vec![]);
            self.sk_w(&format!("::remove {SKILL_VEC}"), vec![])?;
        }
        if !self.sk_has(SKILL_VEC) {
            self.sk_w(&format!("{{:create {SKILL_VEC} {{scope: String, name: String, n: Int => vec: <F32; {}>}}}}", st.dim), vec![])?;
            self.sk_w(
                &format!(
                    "::hnsw create {SKILL_VEC}:hnsw {{dim: {}, m: 32, dtype: F32, fields: [vec], distance: Cosine, ef_construction: 100}}",
                    st.dim
                ),
                vec![],
            )?;
        }
        self.set_skill_meta("embedder", &json!(name))?;
        self.set_skill_meta("embedder_model", &json!(st.model))?;
        self.set_skill_meta("dim", &json!(st.dim))?;
        // embed what is indexed and has no vector yet
        let r = self.sk_r(
            &format!("?[scope, name, n, text] := *{SKILL_CHUNK}{{scope, name, n, text}}, not *{SKILL_VEC}{{scope, name, n}}"),
            vec![],
        )?;
        let mut done = 0;
        for chunk in r.rows.chunks(64) {
            let texts: Vec<&str> = chunk.iter().map(|x| x[3].get_str().unwrap_or("")).collect();
            let vecs = crate::embed::embed_texts(name, &texts)?;
            let rows: Vec<DataValue> = chunk
                .iter()
                .zip(vecs)
                .map(|(x, v)| {
                    DataValue::List(vec![x[0].clone(), x[1].clone(), x[2].clone(), DataValue::List(v.iter().map(|f| DataValue::from(*f as f64)).collect())])
                })
                .collect();
            done += rows.len();
            self.sk_w(&format!("?[scope, name, n, vec] <- $rows :put {SKILL_VEC} {{scope, name, n => vec}}"), vec![("rows", DataValue::List(rows))])?;
        }
        Ok(json!({"embedder": name, "model": st.model, "dim": st.dim, "embedded": done}))
    }

    /// `init {embedder?}`: create the relations; attach an embedder (the
    /// given one, else the memory store's, else none).
    fn skill_init(&'s self, args: &JsonValue) -> Result<JsonValue> {
        self.skill_ensure()?;
        let want = arg_str(args, "embedder").map(String::from).or_else(|| {
            if self.skill_meta("embedder").ok().flatten().is_some() {
                return None;
            }
            self.mem_embedder_name().ok().flatten()
        });
        let mut out = json!({"version": SKILL_SCHEMA_VERSION});
        if let Some(e) = want {
            out["embedder"] = self.skill_attach_embedder(&e)?;
        } else if let Some((e, d)) = self.skill_embedder() {
            out["embedder"] = json!({"embedder": e, "dim": d});
        }
        Ok(out)
    }

    /// `config {embedder?, lifecycle?, policy?, learn?}`: set and return the
    /// configuration.
    fn skill_config(&'s self, args: &JsonValue) -> Result<JsonValue> {
        self.skill_ensure()?;
        if let Some(e) = arg_str(args, "embedder") {
            self.skill_attach_embedder(e)?;
        }
        if let Some(l) = args.get("lifecycle").filter(|v| !v.is_null()) {
            let mut cur = serde_json::to_value(self.skill_lifecycle_opts()).into_diagnostic()?;
            merge(&mut cur, l);
            let o: LifecycleOpts = serde_json::from_value(cur).map_err(|e| miette!("skill config lifecycle: {e}"))?;
            self.set_skill_meta("lifecycle", &serde_json::to_value(o).into_diagnostic()?)?;
        }
        if let Some(l) = args.get("policy").filter(|v| !v.is_null()) {
            let mut cur = serde_json::to_value(self.skill_policy()).into_diagnostic()?;
            merge(&mut cur, l);
            let o: PolicyOpts = serde_json::from_value(cur).map_err(|e| miette!("skill config policy: {e}"))?;
            self.set_skill_meta("policy", &serde_json::to_value(o).into_diagnostic()?)?;
        }
        if let Some(l) = args.get("learn").filter(|v| !v.is_null()) {
            self.skill_learn_configure(l)?;
        }
        if args.get("router").and_then(|v| v.as_str()) == Some("reset") {
            self.sk_w(&format!("?[key] <- [['router']] :rm {SKILL_META} {{key}}"), vec![])?;
        }
        let mut learn = self.skill_meta("learn")?.unwrap_or(JsonValue::Null);
        if let Some(l) = learn.get_mut("llm") {
            // the key itself is never stored; only the variable's name
            l.as_object_mut().map(|m| m.remove("api_key"));
        }
        Ok(json!({
            "version": SKILL_SCHEMA_VERSION,
            "embedder": self.skill_meta("embedder")?,
            "lifecycle": self.skill_lifecycle_opts(),
            "policy": self.skill_policy(),
            "learn": learn,
            "router": self.skill_router_weights(),
        }))
    }

    // ---------------------------------------------------------- versions

    /// The catalog row of a skill.
    pub(crate) fn skill_row(&'s self, scope: &str, name: &str) -> Result<Option<serde_json::Map<String, JsonValue>>> {
        let r = self.sk_r(
            &format!("?[scope, name, ver, status, description, created, updated, actor] := *{SKILL}{{scope, name, ver, status, description, created, updated, actor}}, scope = $s, name = $n"),
            vec![("s", sv(scope)), ("n", sv(name))],
        )?;
        Ok(rows_json(&r).into_iter().next())
    }

    /// A version row, without its file bytes.
    pub(crate) fn skill_ver(&'s self, ver: &str) -> Result<Option<serde_json::Map<String, JsonValue>>> {
        let r = self.sk_r(
            &format!(
                "?[ver, scope, name, seq, digest, frontmatter, body, files, parent, origin, edits, actor, provenance, scan, created, status] := \
                 *{SKILL_VER}{{ver, scope, name, seq, digest, frontmatter, body, files, parent, origin, edits, actor, provenance, scan, created}}, ver = $v, \
                 *{SKILL_VSTAT}{{ver, status}}"
            ),
            vec![("v", sv(ver))],
        )?;
        Ok(rows_json(&r).into_iter().next())
    }

    /// Resolve `{ver}` or `{name, scope?}` (the live version, else the newest).
    pub(crate) fn skill_resolve(&'s self, args: &JsonValue) -> Result<serde_json::Map<String, JsonValue>> {
        if let Some(v) = arg_str(args, "ver").or_else(|| arg_str(args, "version")) {
            return self.skill_ver(v)?.ok_or_else(|| miette!("no skill version '{v}'"));
        }
        let name = arg_str(args, "name").or_else(|| arg_str(args, "skill")).ok_or_else(|| miette!("needs 'name' (or 'ver')"))?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        if let Some(row) = self.skill_row(&scope, name)? {
            if let Some(v) = row.get("ver").and_then(|v| v.as_str()) {
                return self.skill_ver(v)?.ok_or_else(|| miette!("skill '{name}' points at a missing version '{v}'"));
            }
        }
        let r = self.sk_r(
            &format!("?[ver, seq] := *{SKILL_VER}{{ver, scope: $s, name: $n, seq}} :order -seq :limit 1"),
            vec![("s", sv(&scope)), ("n", sv(name))],
        )?;
        let v = r.rows.first().and_then(|x| cs(&x[0])).ok_or_else(|| miette!("no skill '{name}' in scope '{scope}'"))?;
        self.skill_ver(&v)?.ok_or_else(|| miette!("no skill version '{v}'"))
    }

    /// The bytes of one file of a version, checked against its digest.
    pub(crate) fn skill_file(&'s self, files: &JsonValue, path: &str) -> Result<Vec<u8>> {
        let e = entries_of(files).into_iter().find(|f| f.path == path).ok_or_else(|| miette!("the skill has no file '{path}'"))?;
        let r = self.sk_r(&format!("?[bytes] := *{SKILL_BLOB}{{digest: $d, bytes}}"), vec![("d", sv(&e.digest))])?;
        let b = match r.rows.first().map(|x| &x[0]) {
            Some(DataValue::Bytes(b)) => b.clone(),
            _ => bail!("the content of '{path}' ({}) is missing", e.digest),
        };
        if digest_of(&b) != e.digest || b.len() != e.size {
            bail!("the content of '{path}' does not match its digest");
        }
        Ok(b)
    }

    /// Every file of a version.
    pub(crate) fn skill_files(&'s self, files: &JsonValue) -> Result<BTreeMap<String, Vec<u8>>> {
        entries_of(files).iter().map(|e| Ok((e.path.clone(), self.skill_file(files, &e.path)?))).collect()
    }

    /// Store a new version of a skill from its files (`SKILL.md` required).
    /// Identical content in the same scope returns the existing version. A
    /// version the scanner flags is quarantined; anything else starts as a
    /// candidate.
    pub(crate) fn skill_put_files(
        &'s self,
        scope: &str,
        files: BTreeMap<String, Vec<u8>>,
        origin: &str,
        parent: Option<&str>,
        provenance: JsonValue,
        actor: &JsonValue,
        edits: Option<JsonValue>,
    ) -> Result<JsonValue> {
        self.skill_ensure()?;
        for p in files.keys() {
            check_path(p)?;
        }
        let md = files.get("SKILL.md").ok_or_else(|| miette!("a skill needs a SKILL.md"))?;
        let text = std::str::from_utf8(md).map_err(|_| miette!("SKILL.md is not UTF-8"))?;
        let doc = parse_skill_md(text)?;
        let total: usize = files.values().map(|b| b.len()).sum();
        if files.len() > MAX_FILES || total > MAX_TOTAL_BYTES {
            bail!("skill '{}' is over the SEP-2640 limits ({MAX_FILES} files, 16 MiB)", doc.name);
        }
        let (entries, digest) = manifest(&files);
        let existing = self.sk_r(
            &format!("?[ver] := *{SKILL_VER}{{ver, scope: $s, name: $n, digest: $d}}"),
            vec![("s", sv(scope)), ("n", sv(&doc.name)), ("d", sv(&digest))],
        )?;
        if let Some(v) = existing.rows.first().and_then(|x| cs(&x[0])) {
            let row = self.skill_ver(&v)?.unwrap_or_default();
            return Ok(json!({"ver": v, "name": doc.name, "scope": scope, "digest": digest, "status": row.get("status"), "existing": true}));
        }
        let scan = crate::skill_scan::scan(&files, &doc);
        let quarantine = scan.get("verdict").and_then(|v| v.as_str()) == Some("quarantine");
        let seq = self
            .sk_r(&format!("?[max(seq)] := *{SKILL_VER}{{scope: $s, name: $n, seq}}"), vec![("s", sv(scope)), ("n", sv(&doc.name))])?
            .rows
            .first()
            .and_then(|x| x[0].get_int())
            .unwrap_or(0)
            + 1;
        let ver = new_id("sv");
        let now = now_us();
        let mut fm = doc.frontmatter.clone();
        if !fm.is_object() {
            fm = json!({});
        }
        let mut prov = if provenance.is_object() { provenance } else { json!({}) };
        prov["permissions"] = permissions_of(&doc.frontmatter);
        let status = if quarantine { "quarantined" } else { "candidate" };
        let blobs: Vec<DataValue> = files.iter().map(|(_, b)| DataValue::List(vec![sv(&digest_of(b)), DataValue::Bytes(b.clone())])).collect();
        let script = format!(
            "{{?[digest, bytes] <- $blobs :put {SKILL_BLOB} {{digest => bytes}}}} \
             {{?[ver, scope, name, seq, digest, frontmatter, body, files, parent, origin, edits, actor, provenance, scan, created] <- \
                [[$ver, $scope, $name, $seq, $digest, $fm, $body, $files, $parent, $origin, $edits, $actor, $prov, $scan, $now]] \
              :put {SKILL_VER} {{ver => scope, name, seq, digest, frontmatter, body, files, parent, origin, edits, actor, provenance, scan, created}}}} \
             {{?[ver, status, at, reason] <- [[$ver, $status, $now, $reason]] :put {SKILL_VSTAT} {{ver => status, at, reason}}}} \
             {{?[scope, name, at, eid, op, ver, from, actor, detail] <- [[$scope, $name, $now, $eid, 'put', $ver, $parent, $actor, $detail]] \
              :put {SKILL_LOG} {{scope, name, at, eid => op, ver, from, actor, detail}}}}"
        );
        self.sk_w(
            &script,
            vec![
                ("blobs", DataValue::List(blobs)),
                ("ver", sv(&ver)),
                ("scope", sv(scope)),
                ("name", sv(&doc.name)),
                ("seq", DataValue::from(seq)),
                ("digest", sv(&digest)),
                ("fm", jdn(&fm)),
                ("body", sv(&doc.body)),
                ("files", jdn(&manifest_json(&entries))),
                ("parent", osv(parent)),
                ("origin", sv(origin)),
                ("edits", edits.as_ref().map(jd).unwrap_or(DataValue::Null)),
                ("actor", jdn(actor)),
                ("prov", jdn(&prov)),
                ("scan", jdn(&scan)),
                ("now", DataValue::from(now)),
                ("status", sv(status)),
                ("reason", if quarantine { sv("scan") } else { DataValue::Null }),
                ("eid", sv(&new_id("se"))),
                ("detail", jdn(&json!({"origin": origin, "digest": digest, "status": status}))),
            ],
        )?;
        // a skill seen for the first time gets a catalog row (no live version)
        if self.skill_row(scope, &doc.name)?.is_none() {
            self.sk_w(
                &format!(
                    "?[scope, name, ver, status, description, created, updated, actor] <- [[$s, $n, null, $st, $d, $now, $now, $a]] \
                     :put {SKILL} {{scope, name => ver, status, description, created, updated, actor}}"
                ),
                vec![
                    ("s", sv(scope)),
                    ("n", sv(&doc.name)),
                    ("st", sv(if quarantine { "quarantined" } else { "candidate" })),
                    ("d", sv(&doc.description)),
                    ("now", DataValue::from(now)),
                    ("a", jdn(actor)),
                ],
            )?;
        }
        Ok(json!({"ver": ver, "name": doc.name, "scope": scope, "seq": seq, "digest": digest, "status": status, "scan": scan}))
    }

    /// `put {skill_md, files?, scope?, origin?, parent?, provenance?, activate?}`.
    fn skill_put(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let scope = norm_scope(arg_str(args, "scope"))?;
        let actor = actor_of(args.get("actor"));
        let md = arg_str(args, "skill_md").ok_or_else(|| miette!("skill put needs 'skill_md' (the SKILL.md text)"))?;
        let mut files = BTreeMap::new();
        files.insert("SKILL.md".to_string(), md.as_bytes().to_vec());
        if let Some(fs) = args.get("files").and_then(|f| f.as_object()) {
            for (p, v) in fs {
                if p == "SKILL.md" {
                    continue;
                }
                let b = match v {
                    JsonValue::String(s) => s.as_bytes().to_vec(),
                    JsonValue::Object(o) if o.contains_key("b64") => base64::engine::general_purpose::STANDARD
                        .decode(o["b64"].as_str().unwrap_or(""))
                        .map_err(|e| miette!("skill put: file '{p}' is not valid base64: {e}"))?,
                    _ => bail!("skill put: file '{p}' must be text or {{b64}}"),
                };
                files.insert(p.clone(), b);
            }
        }
        let origin = arg_str(args, "origin").unwrap_or("authored");
        let prov = args.get("provenance").cloned().unwrap_or(json!({}));
        let mut out = self.skill_put_files(&scope, files, origin, arg_str(args, "parent"), prov, &actor, args.get("edits").cloned())?;
        if args.get("activate").and_then(|v| v.as_bool()) == Some(true) && out["status"] == "candidate" {
            let v = out["ver"].as_str().unwrap_or_default().to_string();
            out["activated"] = self.skill_activate_ver(&v, &actor, "put")?;
            out["status"] = json!("active");
        }
        Ok(out)
    }

    /// `import {path, scope?, activate?}`: a skill folder, a folder of skill
    /// folders, or a Claude Code plugin.
    fn skill_import(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let path = arg_str(args, "path").ok_or_else(|| miette!("skill import needs 'path'"))?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        let actor = actor_of(args.get("actor"));
        let activate = args.get("activate").and_then(|v| v.as_bool()).unwrap_or(false);
        let mut out = vec![];
        for dir in skill_dirs(std::path::Path::new(path))? {
            let files = read_skill_dir(&dir)?;
            let doc = parse_skill_md(std::str::from_utf8(&files["SKILL.md"]).map_err(|_| miette!("{}: SKILL.md is not UTF-8", dir.display()))?)?;
            let folder = dir.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
            if folder != doc.name {
                bail!("{}: the skill name '{}' must match its folder name '{folder}' (agentskills.io)", dir.display(), doc.name);
            }
            let prov = json!({"source": dir.to_string_lossy(), "imported": now_us()});
            let mut r = self.skill_put_files(&scope, files, "imported", None, prov, &actor, None)?;
            if activate && r["status"] == "candidate" {
                let v = r["ver"].as_str().unwrap_or_default().to_string();
                self.skill_activate_ver(&v, &actor, "import")?;
                r["status"] = json!("active");
            }
            out.push(r);
        }
        Ok(json!({"imported": out}))
    }

    /// `export {path, name? | all, scope?, format: folder|plugin, annotate?}`:
    /// write Agent Skills folders. Without `annotate` the files are the exact
    /// bytes stored (a lossless round trip); `annotate` adds `kleos.*` keys
    /// to the frontmatter `metadata`.
    fn skill_export(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let path = std::path::PathBuf::from(arg_str(args, "path").ok_or_else(|| miette!("skill export needs 'path'"))?);
        let format = arg_str(args, "format").unwrap_or("folder");
        let scope = norm_scope(arg_str(args, "scope"))?;
        let names: Vec<String> = if arg_str(args, "name").is_some() || arg_str(args, "ver").is_some() {
            vec![self.skill_resolve(args)?["name"].as_str().unwrap_or_default().to_string()]
        } else {
            self.sk_r(&format!("?[name] := *{SKILL}{{scope: $s, name, status: 'active'}}"), vec![("s", sv(&scope))])?
                .rows
                .iter()
                .filter_map(|x| cs(&x[0]))
                .collect()
        };
        let root = if format == "plugin" { path.join("skills") } else { path.clone() };
        std::fs::create_dir_all(&root).into_diagnostic()?;
        let mut written = vec![];
        for n in &names {
            let mut a = json!({"name": n, "scope": scope});
            if let Some(v) = arg_str(args, "ver") {
                a = json!({"ver": v});
            }
            let row = self.skill_resolve(&a)?;
            let mut files = self.skill_files(&row["files"])?;
            if args.get("annotate").and_then(|v| v.as_bool()) == Some(true) {
                let stats = self.skill_stats_of(row["ver"].as_str().unwrap_or_default())?;
                files.insert("SKILL.md".into(), annotate_md(&files["SKILL.md"], &row, &stats)?);
            }
            let dir = root.join(n);
            for (p, b) in &files {
                let fp = dir.join(p);
                if let Some(parent) = fp.parent() {
                    std::fs::create_dir_all(parent).into_diagnostic()?;
                }
                std::fs::write(&fp, b).into_diagnostic()?;
            }
            written.push(json!({"name": n, "ver": row["ver"], "digest": row["digest"], "path": dir.to_string_lossy()}));
        }
        if format == "plugin" {
            let pj = path.join(".claude-plugin");
            std::fs::create_dir_all(&pj).into_diagnostic()?;
            let name = arg_str(args, "plugin").unwrap_or("kleos-skills");
            let manifest = json!({"name": name, "version": "0.1.0", "description": format!("Skills exported from kleos (scope '{scope}')"), "skills": "./skills/"});
            std::fs::write(pj.join("plugin.json"), serde_json::to_string_pretty(&manifest).into_diagnostic()?).into_diagnostic()?;
        }
        Ok(json!({"exported": written, "format": format}))
    }

    // ---------------------------------------------------------- lifecycle

    /// The newest head timestamp of a skill, so a new head never lands on the
    /// same microsecond as the last one.
    fn skill_head_at(&'s self, scope: &str, name: &str) -> Result<i64> {
        let last = self.skill_heads(scope, name)?.first().map(|h| h.0).unwrap_or(0);
        Ok(now_us().max(last + 1))
    }

    /// The head timeline, newest first: `(at, ver, status, reason)`.
    pub(crate) fn skill_heads(&'s self, scope: &str, name: &str) -> Result<Vec<(i64, Option<String>, String, Option<String>, JsonValue)>> {
        let r = self.sk_r(
            &format!("?[vt, ver, status, reason, actor] := *{SKILL_HEAD}{{scope: $s, name: $n, vt, ver, status, reason, actor}}"),
            vec![("s", sv(scope)), ("n", sv(name))],
        )?;
        let mut out: Vec<_> = r
            .rows
            .iter()
            .filter_map(|x| match &x[0] {
                DataValue::Validity(v) if v.is_assert.0 => Some((v.timestamp.0 .0, cs(&x[1]), cs(&x[2]).unwrap_or_default(), cs(&x[3]), jv(&x[4]))),
                _ => None,
            })
            .collect();
        out.sort_by(|a, b| b.0.cmp(&a.0));
        Ok(out)
    }

    /// Index rows (chunks, and vectors when an embedder is attached) of a
    /// version, as script parameters.
    fn skill_index_rows(&'s self, row: &serde_json::Map<String, JsonValue>) -> Result<(Vec<DataValue>, Option<Vec<DataValue>>)> {
        let scope = row["scope"].as_str().unwrap_or_default();
        let name = row["name"].as_str().unwrap_or_default();
        let ver = row["ver"].as_str().unwrap_or_default();
        let desc = row["frontmatter"].get("description").and_then(|d| d.as_str()).unwrap_or_default();
        let mut refs = vec![];
        for e in entries_of(&row["files"]) {
            if file_kind(&e.path) == "reference" && (e.path.ends_with(".md") || e.path.ends_with(".txt")) && e.size < 200_000 {
                if let Ok(b) = self.skill_file(&row["files"], &e.path) {
                    refs.push((e.path.clone(), String::from_utf8_lossy(&b).into_owned()));
                }
            }
        }
        let chunks = chunks_of(name, desc, row["body"].as_str().unwrap_or_default(), &refs);
        let crow: Vec<DataValue> = chunks
            .iter()
            .enumerate()
            .map(|(i, (k, t))| DataValue::List(vec![sv(scope), sv(name), DataValue::from(i as i64), sv(ver), sv(k), sv(t)]))
            .collect();
        let vrows = match self.skill_embedder() {
            Some((e, _)) => {
                let texts: Vec<&str> = chunks.iter().map(|(_, t)| t.as_str()).collect();
                let vecs = crate::embed::embed_texts(&e, &texts)?;
                Some(
                    vecs.iter()
                        .enumerate()
                        .map(|(i, v)| {
                            DataValue::List(vec![sv(scope), sv(name), DataValue::from(i as i64), DataValue::List(v.iter().map(|f| DataValue::from(*f as f64)).collect())])
                        })
                        .collect(),
                )
            }
            None => None,
        };
        Ok((crow, vrows))
    }

    /// Assert a new head for `(scope, name)` in one transaction with the
    /// catalog, the version statuses, the index and the log. `ver: None`
    /// takes the skill out of service (`status`: retired or quarantined).
    #[allow(clippy::too_many_arguments)]
    fn skill_set_head(
        &'s self,
        scope: &str,
        name: &str,
        ver: Option<&serde_json::Map<String, JsonValue>>,
        status: &str,
        old_status: Option<&str>,
        actor: &JsonValue,
        op: &str,
        reason: Option<&str>,
    ) -> Result<JsonValue> {
        let at = self.skill_head_at(scope, name)?;
        let cur = self.skill_row(scope, name)?;
        let cur_ver = cur.as_ref().and_then(|r| r.get("ver")).and_then(|v| v.as_str()).map(String::from);
        let new_ver = ver.and_then(|r| r["ver"].as_str()).map(String::from);
        let desc = ver
            .and_then(|r| r["frontmatter"].get("description").and_then(|d| d.as_str()))
            .map(String::from)
            .or_else(|| cur.as_ref().and_then(|r| r.get("description")).and_then(|d| d.as_str()).map(String::from))
            .unwrap_or_default();
        let created = cur.as_ref().and_then(|r| r.get("created")).and_then(|v| v.as_i64()).unwrap_or(at);
        let mut script = format!(
            "{{?[scope, name, vt, ver, status, actor, reason] <- [[$s, $n, [$at, true], $v, $st, $a, $why]] \
              :put {SKILL_HEAD} {{scope, name, vt => ver, status, actor, reason}}}} \
             {{?[scope, name, ver, status, description, created, updated, actor] <- [[$s, $n, $v, $st, $d, $created, $at, $a]] \
              :put {SKILL} {{scope, name => ver, status, description, created, updated, actor}}}} \
             {{?[scope, name, n] := *{SKILL_CHUNK}{{scope, name, n}}, scope = $s, name = $n :rm {SKILL_CHUNK} {{scope, name, n}}}} \
             {{?[scope, name, at, eid, op, ver, from, actor, detail] <- [[$s, $n, $at, $eid, $op, $v, $from, $a, $detail]] \
              :put {SKILL_LOG} {{scope, name, at, eid => op, ver, from, actor, detail}}}}"
        );
        let mut p = vec![
            ("s", sv(scope)),
            ("n", sv(name)),
            ("at", DataValue::from(at)),
            ("v", osv(new_ver.as_deref())),
            ("st", sv(status)),
            ("a", jdn(actor)),
            ("why", osv(reason)),
            ("d", sv(&desc)),
            ("created", DataValue::from(created)),
            ("eid", sv(&new_id("se"))),
            ("op", sv(op)),
            ("from", osv(cur_ver.as_deref())),
            ("detail", jdn(&json!({"status": status, "reason": reason}))),
        ];
        if self.sk_has(SKILL_VEC) {
            script.push_str(&format!(
                "{{?[scope, name, n] := *{SKILL_VEC}{{scope, name, n}}, scope = $s, name = $n :rm {SKILL_VEC} {{scope, name, n}}}}"
            ));
        }
        if let (Some(c), Some(os)) = (&cur_ver, old_status) {
            if Some(c) != new_ver.as_ref() {
                script.push_str(&format!("{{?[ver, status, at, reason] <- [[$cv, $os, $at, $why]] :put {SKILL_VSTAT} {{ver => status, at, reason}}}}"));
                p.push(("cv", sv(c)));
                p.push(("os", sv(os)));
            }
        }
        if let Some(r) = ver {
            script.push_str(&format!("{{?[ver, status, at, reason] <- [[$v, 'active', $at, $why]] :put {SKILL_VSTAT} {{ver => status, at, reason}}}}"));
            if status == "active" {
                let (crow, vrows) = self.skill_index_rows(r)?;
                script.push_str(&format!("{{?[scope, name, n, ver, kind, text] <- $chunks :put {SKILL_CHUNK} {{scope, name, n => ver, kind, text}}}}"));
                p.push(("chunks", DataValue::List(crow)));
                if let (Some(v), true) = (vrows, self.sk_has(SKILL_VEC)) {
                    script.push_str(&format!("{{?[scope, name, n, vec] <- $vecs :put {SKILL_VEC} {{scope, name, n => vec}}}}"));
                    p.push(("vecs", DataValue::List(v)));
                }
            }
        }
        self.sk_w(&script, p)?;
        Ok(json!({"scope": scope, "name": name, "ver": new_ver, "previous": cur_ver, "status": status, "at": at}))
    }

    /// Make a version the live one of its skill. Quarantined and rejected
    /// versions cannot be activated.
    pub(crate) fn skill_activate_ver(&'s self, ver: &str, actor: &JsonValue, why: &str) -> Result<JsonValue> {
        let row = self.skill_ver(ver)?.ok_or_else(|| miette!("no skill version '{ver}'"))?;
        let st = row["status"].as_str().unwrap_or_default();
        if st == "quarantined" || st == "rejected" {
            bail!("skill version '{ver}' is {st}; it cannot be activated (see `::skill scan` / `release`)");
        }
        let scope = row["scope"].as_str().unwrap_or_default().to_string();
        let name = row["name"].as_str().unwrap_or_default().to_string();
        self.skill_set_head(&scope, &name, Some(&row), "active", Some("superseded"), actor, "activate", Some(why))
    }

    /// `activate {ver | name+scope}`: the newest candidate when only a name is given.
    fn skill_activate(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let actor = actor_of(args.get("actor"));
        let ver = match arg_str(args, "ver").or_else(|| arg_str(args, "version")) {
            Some(v) => v.to_string(),
            None => {
                let name = arg_str(args, "name").ok_or_else(|| miette!("skill activate needs 'ver' or 'name'"))?;
                let scope = norm_scope(arg_str(args, "scope"))?;
                self.sk_r(
                    &format!("?[ver, seq] := *{SKILL_VER}{{ver, scope: $s, name: $n, seq}}, *{SKILL_VSTAT}{{ver, status: 'candidate'}} :order -seq :limit 1"),
                    vec![("s", sv(&scope)), ("n", sv(name))],
                )?
                .rows
                .first()
                .and_then(|x| cs(&x[0]))
                .ok_or_else(|| miette!("skill '{name}' has no candidate version to activate in scope '{scope}'"))?
            }
        };
        self.skill_activate_ver(&ver, &actor, arg_str(args, "reason").unwrap_or("manual"))
    }

    /// `rollback {name, scope?, to?, reason?}`: back to the previous live
    /// version (or `to`); the current one is retired.
    pub(crate) fn skill_rollback(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let name = arg_str(args, "name").ok_or_else(|| miette!("skill rollback needs 'name'"))?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        let actor = actor_of(args.get("actor"));
        let cur = self.skill_row(&scope, name)?.and_then(|r| r.get("ver").and_then(|v| v.as_str()).map(String::from));
        let target = match arg_str(args, "to") {
            Some(t) => t.to_string(),
            None => {
                let heads = self.skill_heads(&scope, name)?;
                let mut found = None;
                for (_, v, _, _, _) in heads.iter() {
                    if let Some(v) = v {
                        if Some(v) != cur.as_ref() {
                            let st = self.skill_ver(v)?.and_then(|r| r.get("status").and_then(|s| s.as_str()).map(String::from));
                            if !matches!(st.as_deref(), Some("quarantined" | "rejected")) {
                                found = Some(v.clone());
                                break;
                            }
                        }
                    }
                }
                found.ok_or_else(|| miette!("skill '{name}' has no earlier version to roll back to"))?
            }
        };
        let row = self.skill_ver(&target)?.ok_or_else(|| miette!("no skill version '{target}'"))?;
        let reason = arg_str(args, "reason").unwrap_or("rollback");
        self.skill_set_head(&scope, name, Some(&row), "active", Some("retired"), &actor, "rollback", Some(reason))
    }

    /// `undo {name, scope?}`: one step back in the head timeline (an
    /// activation, rollback, retirement or quarantine is reverted).
    fn skill_undo(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let name = arg_str(args, "name").ok_or_else(|| miette!("skill undo needs 'name'"))?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        let actor = actor_of(args.get("actor"));
        let heads = self.skill_heads(&scope, name)?;
        let Some(prev) = heads.get(1) else { bail!("skill '{name}' has nothing to undo") };
        match &prev.1 {
            Some(v) => {
                let row = self.skill_ver(v)?.ok_or_else(|| miette!("no skill version '{v}'"))?;
                if row["status"] == "quarantined" {
                    bail!("the previous version '{v}' is quarantined; undo would serve it");
                }
                self.skill_set_head(&scope, name, Some(&row), "active", Some("candidate"), &actor, "undo", Some("undo"))
            }
            None => self.skill_set_head(&scope, name, None, &prev.2.clone(), Some("candidate"), &actor, "undo", Some("undo")),
        }
    }

    /// `retire {name, scope?, reason?}`.
    pub(crate) fn skill_retire(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let name = arg_str(args, "name").ok_or_else(|| miette!("skill retire needs 'name'"))?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        let actor = actor_of(args.get("actor"));
        if self.skill_row(&scope, name)?.is_none() {
            bail!("no skill '{name}' in scope '{scope}'");
        }
        self.skill_set_head(&scope, name, None, "retired", Some("retired"), &actor, "retire", Some(arg_str(args, "reason").unwrap_or("manual")))
    }

    /// `quarantine {ver, reason?}`: never served again; if it is live the skill
    /// goes out of service.
    fn skill_quarantine(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let row = self.skill_resolve(args)?;
        let ver = row["ver"].as_str().unwrap_or_default().to_string();
        let scope = row["scope"].as_str().unwrap_or_default().to_string();
        let name = row["name"].as_str().unwrap_or_default().to_string();
        let actor = actor_of(args.get("actor"));
        let reason = arg_str(args, "reason").unwrap_or("manual");
        let live = self.skill_row(&scope, &name)?.and_then(|r| r.get("ver").and_then(|v| v.as_str()).map(String::from));
        if live.as_deref() == Some(ver.as_str()) {
            self.skill_set_head(&scope, &name, None, "quarantined", Some("quarantined"), &actor, "quarantine", Some(reason))?;
        }
        self.sk_w(
            &format!("?[ver, status, at, reason] <- [[$v, 'quarantined', $now, $r]] :put {SKILL_VSTAT} {{ver => status, at, reason}}"),
            vec![("v", sv(&ver)), ("now", DataValue::from(now_us())), ("r", sv(reason))],
        )?;
        Ok(json!({"ver": ver, "status": "quarantined"}))
    }

    /// `release {ver, confirm: true}`: a quarantined version back to
    /// candidate (an explicit override of the scanner).
    fn skill_release(&'s self, args: &JsonValue) -> Result<JsonValue> {
        if args.get("confirm").and_then(|v| v.as_bool()) != Some(true) {
            bail!("skill release overrides the security scan; pass confirm: true after review");
        }
        let row = self.skill_resolve(args)?;
        let ver = row["ver"].as_str().unwrap_or_default().to_string();
        if row["status"] != "quarantined" {
            bail!("skill version '{ver}' is not quarantined");
        }
        let actor = actor_of(args.get("actor"));
        self.sk_w(
            &format!(
                "{{?[ver, status, at, reason] <- [[$v, 'candidate', $now, 'released']] :put {SKILL_VSTAT} {{ver => status, at, reason}}}} \
                 {{?[scope, name, at, eid, op, ver, from, actor, detail] <- [[$s, $n, $now, $eid, 'release', $v, null, $a, null]] \
                  :put {SKILL_LOG} {{scope, name, at, eid => op, ver, from, actor, detail}}}}"
            ),
            vec![
                ("v", sv(&ver)),
                ("s", sv(row["scope"].as_str().unwrap_or_default())),
                ("n", sv(row["name"].as_str().unwrap_or_default())),
                ("now", DataValue::from(now_us())),
                ("eid", sv(&new_id("se"))),
                ("a", jdn(&actor)),
            ],
        )?;
        Ok(json!({"ver": ver, "status": "candidate"}))
    }

    /// `erase {name, scope?, confirm: true}`: delete a skill and every version,
    /// outcome and index row (blobs other skills share are kept).
    fn skill_erase(&'s self, args: &JsonValue) -> Result<JsonValue> {
        if args.get("confirm").and_then(|v| v.as_bool()) != Some(true) {
            bail!("skill erase deletes every version and its history; pass confirm: true");
        }
        let name = arg_str(args, "name").ok_or_else(|| miette!("skill erase needs 'name'"))?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        let vers: Vec<String> = self
            .sk_r(&format!("?[ver] := *{SKILL_VER}{{ver, scope: $s, name: $n}}"), vec![("s", sv(&scope)), ("n", sv(name))])?
            .rows
            .iter()
            .filter_map(|x| cs(&x[0]))
            .collect();
        let mut script = format!(
            "{{?[ver] := ver in $vers :rm {SKILL_VER} {{ver}}}} \
             {{?[ver] := ver in $vers :rm {SKILL_VSTAT} {{ver}}}} \
             {{?[ver, oid] := ver in $vers, *{SKILL_OUTCOME}{{ver, oid}} :rm {SKILL_OUTCOME} {{ver, oid}}}} \
             {{?[ver, kid] := ver in $vers, *{SKILL_SIG}{{ver, kid}} :rm {SKILL_SIG} {{ver, kid}}}} \
             {{?[scope, name, vt] := *{SKILL_HEAD}{{scope, name, vt}}, scope = $s, name = $n :rm {SKILL_HEAD} {{scope, name, vt}}}} \
             {{?[scope, name] <- [[$s, $n]] :rm {SKILL} {{scope, name}}}} \
             {{?[scope, name, n] := *{SKILL_CHUNK}{{scope, name, n}}, scope = $s, name = $n :rm {SKILL_CHUNK} {{scope, name, n}}}} \
             {{?[scope, name, case] := *{SKILL_EVAL}{{scope, name, case}}, scope = $s, name = $n :rm {SKILL_EVAL} {{scope, name, case}}}}"
        );
        if self.sk_has(SKILL_VEC) {
            script.push_str(&format!(
                "{{?[scope, name, n] := *{SKILL_VEC}{{scope, name, n}}, scope = $s, name = $n :rm {SKILL_VEC} {{scope, name, n}}}}"
            ));
        }
        self.sk_w(&script, vec![("vers", str_list(&vers)), ("s", sv(&scope)), ("n", sv(name))])?;
        self.skill_gc_blobs()?;
        Ok(json!({"name": name, "scope": scope, "erased_versions": vers.len()}))
    }

    /// Drop blobs no version lists any more.
    fn skill_gc_blobs(&'s self) -> Result<usize> {
        let mut used = BTreeSet::new();
        for r in self.sk_r(&format!("?[files] := *{SKILL_VER}{{files}}"), vec![])?.rows {
            for e in entries_of(&jv(&r[0])) {
                used.insert(e.digest);
            }
        }
        let all: Vec<String> = self.sk_r(&format!("?[digest] := *{SKILL_BLOB}{{digest}}"), vec![])?.rows.iter().filter_map(|x| cs(&x[0])).collect();
        let gone: Vec<String> = all.into_iter().filter(|d| !used.contains(d)).collect();
        if !gone.is_empty() {
            self.sk_w(&format!("?[digest] := digest in $g :rm {SKILL_BLOB} {{digest}}"), vec![("g", str_list(&gone))])?;
        }
        Ok(gone.len())
    }

    // ---------------------------------------------------------- reads

    /// `get {name+scope | ver, files?: bool, file?: path}`.
    fn skill_get(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let row = self.skill_resolve(args)?;
        let scope = row["scope"].as_str().unwrap_or_default().to_string();
        let name = row["name"].as_str().unwrap_or_default().to_string();
        if let Some(p) = arg_str(args, "file") {
            let b = self.skill_file(&row["files"], p)?;
            let e = entries_of(&row["files"]).into_iter().find(|f| f.path == p).unwrap();
            return Ok(match String::from_utf8(b.clone()) {
                Ok(t) => json!({"path": p, "uri": skill_uri(&scope, &name, p), "digest": e.digest, "size": e.size, "text": t}),
                Err(_) => json!({"path": p, "uri": skill_uri(&scope, &name, p), "digest": e.digest, "size": e.size,
                                  "b64": base64::engine::general_purpose::STANDARD.encode(&b)}),
            });
        }
        let ver = row["ver"].as_str().unwrap_or_default().to_string();
        let mut out = serde_json::Map::from_iter(row.clone());
        out.insert("uri".into(), json!(skill_uri(&scope, &name, "SKILL.md")));
        out.insert("live".into(), json!(self.skill_row(&scope, &name)?.and_then(|r| r.get("ver").cloned()) == Some(json!(ver))));
        out.insert("stats".into(), self.skill_stats_of(&ver)?);
        out.insert("signatures".into(), self.skill_sigs(&ver)?);
        out.insert("resources".into(), JsonValue::Array(
            entries_of(&row["files"]).iter().map(|e| json!({"path": e.path, "uri": skill_uri(&scope, &name, &e.path), "digest": e.digest, "size": e.size})).collect(),
        ));
        if args.get("files").and_then(|v| v.as_bool()) == Some(true) {
            let mut fs = serde_json::Map::new();
            for (p, b) in self.skill_files(&row["files"])? {
                fs.insert(p, match String::from_utf8(b.clone()) {
                    Ok(t) => json!(t),
                    Err(_) => json!({"b64": base64::engine::general_purpose::STANDARD.encode(&b)}),
                });
            }
            out.insert("file_contents".into(), JsonValue::Object(fs));
        }
        Ok(JsonValue::Object(out))
    }

    /// `list {scope?, scopes?, status?, limit?}`: catalog rows with live stats.
    fn skill_list(&'s self, args: &JsonValue) -> Result<JsonValue> {
        self.skill_ensure()?;
        let scopes = self.skill_scopes(args)?;
        let status = arg_str(args, "status");
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
        let r = self.sk_r(
            &format!("?[scope, name, ver, status, description, updated] := *{SKILL}{{scope, name, ver, status, description, updated}}, is_in(scope, $scopes)"),
            vec![("scopes", str_list(&scopes))],
        )?;
        let mut out = vec![];
        for mut o in rows_json(&r) {
            if status.is_some_and(|s| o.get("status").and_then(|x| x.as_str()) != Some(s)) {
                continue;
            }
            if let Some(v) = o.get("ver").and_then(|v| v.as_str()).map(String::from) {
                o.insert("stats".into(), self.skill_stats_of(&v)?);
            }
            let (s, n) = (o["scope"].as_str().unwrap_or_default().to_string(), o["name"].as_str().unwrap_or_default().to_string());
            o.insert("uri".into(), json!(skill_uri(&s, &n, "SKILL.md")));
            out.push(JsonValue::Object(o));
            if out.len() >= limit {
                break;
            }
        }
        Ok(JsonValue::Array(out))
    }

    /// The scopes a call sees: `scopes`, else `[scope]`, narrowest first.
    pub(crate) fn skill_scopes(&'s self, args: &JsonValue) -> Result<Vec<String>> {
        let mut v: Vec<String> = match args.get("scopes").and_then(|s| s.as_array()) {
            Some(a) if !a.is_empty() => a.iter().filter_map(|x| x.as_str()).map(|x| norm_scope(Some(x))).collect::<Result<_>>()?,
            _ => vec![norm_scope(arg_str(args, "scope"))?],
        };
        v.sort_by_key(|s| scope_rank(s));
        v.dedup();
        Ok(v)
    }

    /// `history {name, scope?}`: versions, the head timeline and the log.
    fn skill_history(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let name = arg_str(args, "name").ok_or_else(|| miette!("skill history needs 'name'"))?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        let vers = self.sk_r(
            &format!(
                "?[ver, seq, digest, parent, origin, created, status, reason] := *{SKILL_VER}{{ver, scope: $s, name: $n, seq, digest, parent, origin, created}}, \
                 *{SKILL_VSTAT}{{ver, status, reason}} :order seq"
            ),
            vec![("s", sv(&scope)), ("n", sv(name))],
        )?;
        let heads: Vec<JsonValue> = self
            .skill_heads(&scope, name)?
            .into_iter()
            .map(|(at, v, st, why, a)| json!({"at": at, "ver": v, "status": st, "reason": why, "actor": a}))
            .collect();
        let log = self.sk_r(
            &format!("?[at, op, ver, from, actor, detail] := *{SKILL_LOG}{{scope: $s, name: $n, at, op, ver, from, actor, detail}} :order at"),
            vec![("s", sv(&scope)), ("n", sv(name))],
        )?;
        let mut versions = rows_json(&vers);
        for v in versions.iter_mut() {
            let id = v["ver"].as_str().unwrap_or_default().to_string();
            v.insert("stats".into(), self.skill_stats_of(&id)?);
        }
        Ok(json!({"name": name, "scope": scope, "versions": versions, "heads": heads, "log": rows_json(&log)}))
    }

    /// `as_of {name, scope?, at}`: the version that was live at `at` (µs).
    fn skill_as_of(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let name = arg_str(args, "name").ok_or_else(|| miette!("skill as_of needs 'name'"))?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        let at = args.get("at").and_then(|v| v.as_i64()).ok_or_else(|| miette!("skill as_of needs 'at' (µs)"))?;
        let r = self.sk_r(
            &format!("?[ver, status] := *{SKILL_HEAD}{{scope: $s, name: $n, ver, status @ $at}}"),
            vec![("s", sv(&scope)), ("n", sv(name)), ("at", DataValue::from(at))],
        )?;
        Ok(r.rows.first().map(|x| json!({"ver": jv(&x[0]), "status": jv(&x[1]), "at": at})).unwrap_or(json!({"ver": null, "at": at})))
    }

    // ---------------------------------------------------------- outcomes

    /// Outcome counts of versions: `ver → (uses, successes, latency sum and
    /// count, tokens sum and count, cost sum, last)`.
    pub(crate) fn skill_outcome_counts(&'s self, vers: &[String]) -> Result<BTreeMap<String, OutcomeAgg>> {
        let mut out: BTreeMap<String, OutcomeAgg> = BTreeMap::new();
        if vers.is_empty() || !self.sk_has(SKILL_OUTCOME) {
            return Ok(out);
        }
        let r = self.sk_r(
            &format!("?[ver, success, latency_ms, tokens, cost, at, oid] := ver in $vers, *{SKILL_OUTCOME}{{ver, oid, success, latency_ms, tokens, cost, at}}"),
            vec![("vers", str_list(vers))],
        )?;
        for x in r.rows {
            let Some(v) = cs(&x[0]) else { continue };
            let a = out.entry(v).or_default();
            a.uses += 1;
            if x[1].get_bool() == Some(true) {
                a.successes += 1;
            }
            if let Some(l) = x[2].get_float() {
                a.lat_sum += l;
                a.lat_n += 1;
            }
            if let Some(t) = x[3].get_int() {
                a.tok_sum += t;
                a.tok_n += 1;
            }
            if let Some(c) = x[4].get_float() {
                a.cost += c;
            }
            a.last = a.last.max(x[5].get_int().unwrap_or(0));
        }
        Ok(out)
    }

    /// Stats of one version from its outcomes.
    pub(crate) fn skill_stats_of(&'s self, ver: &str) -> Result<JsonValue> {
        let m = self.skill_outcome_counts(&[ver.to_string()])?;
        Ok(m.get(ver).cloned().unwrap_or_default().to_json())
    }

    /// `stats {ver | name+scope}`: the version's stats, and its predecessor's.
    fn skill_stats(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let row = self.skill_resolve(args)?;
        let ver = row["ver"].as_str().unwrap_or_default().to_string();
        let mut out = json!({"ver": ver, "name": row["name"], "scope": row["scope"], "status": row["status"], "stats": self.skill_stats_of(&ver)?});
        if let Some(p) = self.skill_predecessor(&row)? {
            out["predecessor"] = json!({"ver": p, "stats": self.skill_stats_of(&p)?});
        }
        Ok(out)
    }

    /// The version that was live before this one became live.
    pub(crate) fn skill_predecessor(&'s self, row: &serde_json::Map<String, JsonValue>) -> Result<Option<String>> {
        let ver = row["ver"].as_str().unwrap_or_default();
        let heads = self.skill_heads(row["scope"].as_str().unwrap_or_default(), row["name"].as_str().unwrap_or_default())?;
        let pos = heads.iter().position(|h| h.1.as_deref() == Some(ver));
        Ok(pos.and_then(|i| heads[i + 1..].iter().find_map(|h| h.1.clone().filter(|v| v != ver))))
    }

    /// `record_outcome {ver | name+scope, success, run?, latency_ms?, tokens?,
    /// cost?, model?, task?, metrics?, route?, source?}`. Feeds the stats, the
    /// learned router (with `route`) and the automatic rollback check.
    pub(crate) fn skill_record_outcome(&'s self, args: &JsonValue) -> Result<JsonValue> {
        self.skill_ensure()?;
        let row = self.skill_resolve(args)?;
        let success = args.get("success").and_then(|v| v.as_bool()).ok_or_else(|| miette!("skill record_outcome needs 'success' (bool)"))?;
        let ver = row["ver"].as_str().unwrap_or_default().to_string();
        let oid = new_id("so");
        let f = |k: &str| args.get(k).and_then(|v| v.as_f64()).map(DataValue::from).unwrap_or(DataValue::Null);
        let route = arg_str(args, "route").map(String::from);
        self.sk_w(
            &format!(
                "?[ver, oid, scope, name, at, run, success, latency_ms, tokens, cost, model, task, source, route, actor, metrics] <- \
                 [[$ver, $oid, $s, $n, $now, $run, $ok, $lat, $tok, $cost, $model, $task, $src, $route, $a, $m]] \
                 :put {SKILL_OUTCOME} {{ver, oid => scope, name, at, run, success, latency_ms, tokens, cost, model, task, source, route, actor, metrics}}"
            ),
            vec![
                ("ver", sv(&ver)),
                ("oid", sv(&oid)),
                ("s", sv(row["scope"].as_str().unwrap_or_default())),
                ("n", sv(row["name"].as_str().unwrap_or_default())),
                ("now", DataValue::from(args.get("at").and_then(|v| v.as_i64()).unwrap_or_else(now_us))),
                ("run", osv(arg_str(args, "run"))),
                ("ok", DataValue::from(success)),
                ("lat", f("latency_ms")),
                ("tok", args.get("tokens").and_then(|v| v.as_i64()).map(DataValue::from).unwrap_or(DataValue::Null)),
                ("cost", f("cost")),
                ("model", osv(arg_str(args, "model"))),
                ("task", osv(arg_str(args, "task"))),
                ("src", sv(arg_str(args, "source").unwrap_or("reported"))),
                ("route", osv(route.as_deref())),
                ("a", jdn(&actor_of(args.get("actor")))),
                ("m", args.get("metrics").map(jd).unwrap_or(DataValue::Null)),
            ],
        )?;
        let mut out = json!({"oid": oid, "ver": ver, "success": success});
        if let Some(r) = route {
            if let Ok(u) = self.skill_router_learn(&r, row["scope"].as_str().unwrap_or_default(), row["name"].as_str().unwrap_or_default(), success) {
                out["router"] = u;
            }
        }
        if let Ok(Some(rb)) = self.skill_check_rollback(&row) {
            out["rolled_back"] = rb;
        }
        Ok(out)
    }

    /// Roll back a live version that does worse than its predecessor
    /// (Bayesian comparison of Beta posteriors, minimum samples on both).
    pub(crate) fn skill_check_rollback(&'s self, row: &serde_json::Map<String, JsonValue>) -> Result<Option<JsonValue>> {
        let o = self.skill_lifecycle_opts();
        if !o.auto_rollback {
            return Ok(None);
        }
        let ver = row["ver"].as_str().unwrap_or_default().to_string();
        let (scope, name) = (row["scope"].as_str().unwrap_or_default(), row["name"].as_str().unwrap_or_default());
        let live = self.skill_row(scope, name)?.and_then(|r| r.get("ver").and_then(|v| v.as_str()).map(String::from));
        if live.as_deref() != Some(ver.as_str()) {
            return Ok(None);
        }
        let Some(prev) = self.skill_predecessor(row)? else { return Ok(None) };
        let c = self.skill_outcome_counts(&[ver.clone(), prev.clone()])?;
        let (a, b) = (c.get(&ver).cloned().unwrap_or_default(), c.get(&prev).cloned().unwrap_or_default());
        if a.uses < o.min_samples || b.uses < o.min_samples {
            return Ok(None);
        }
        let p_worse = prob_less(a.successes, a.uses - a.successes, b.successes, b.uses - b.successes);
        if p_worse < o.rollback_confidence {
            return Ok(None);
        }
        let why = format!("auto-rollback: P(worse than {prev}) = {p_worse:.3} ({}/{} vs {}/{})", a.successes, a.uses, b.successes, b.uses);
        let r = self.skill_rollback(&json!({"name": name, "scope": scope, "to": prev, "reason": why}))?;
        Ok(Some(r))
    }

    /// `lifecycle {scope?}`: the maintenance sweep: roll back losing versions,
    /// retire skills unused for `retire_unused_days` and skills that
    /// consistently fail.
    pub(crate) fn skill_lifecycle(&'s self, args: &JsonValue) -> Result<JsonValue> {
        self.skill_ensure()?;
        let o = self.skill_lifecycle_opts();
        let scopes = self.skill_scopes(args)?;
        let all = args.get("scopes").is_none() && arg_str(args, "scope").is_none();
        let r = self.sk_r(
            &format!("?[scope, name, ver, updated] := *{SKILL}{{scope, name, ver, status: 'active', updated}}"),
            vec![],
        )?;
        let now = now_us();
        let (mut rolled, mut retired) = (vec![], vec![]);
        for x in r.rows {
            let (Some(scope), Some(name), Some(ver)) = (cs(&x[0]), cs(&x[1]), cs(&x[2])) else { continue };
            if !all && !scopes.contains(&scope) {
                continue;
            }
            let Some(row) = self.skill_ver(&ver)? else { continue };
            if let Some(rb) = self.skill_check_rollback(&row)? {
                rolled.push(rb);
                continue;
            }
            let st = self.skill_outcome_counts(&[ver.clone()])?.remove(&ver).unwrap_or_default();
            let updated = x[3].get_int().unwrap_or(now);
            let idle_days = (now - st.last.max(updated)) as f64 / 86_400_000_000.0;
            let why = if o.retire_unused_days > 0 && idle_days > o.retire_unused_days as f64 {
                Some(format!("unused for {idle_days:.0} days"))
            } else if st.uses >= o.min_samples && prob_below(st.successes, st.uses - st.successes, o.retire_below) > 0.95 {
                Some(format!("consistently failing: {}/{} successes", st.successes, st.uses))
            } else {
                None
            };
            if let Some(why) = why {
                retired.push(self.skill_retire(&json!({"name": name, "scope": scope, "reason": why}))?);
            }
        }
        Ok(json!({"rolled_back": rolled, "retired": retired}))
    }

    /// `use {run, ver | name+scope, route?}`: a flow run (or any run id) used
    /// this version; its outcome is recorded when the workflow finishes.
    pub(crate) fn skill_use(&'s self, args: &JsonValue) -> Result<JsonValue> {
        self.skill_ensure()?;
        let row = self.skill_resolve(args)?;
        let run = arg_str(args, "run").ok_or_else(|| miette!("skill use needs 'run'"))?;
        self.sk_w(
            &format!("?[run, ver, scope, name, at, route] <- [[$r, $v, $s, $n, $now, $route]] :put {SKILL_USE} {{run, ver => scope, name, at, route}}"),
            vec![
                ("r", sv(run)),
                ("v", sv(row["ver"].as_str().unwrap_or_default())),
                ("s", sv(row["scope"].as_str().unwrap_or_default())),
                ("n", sv(row["name"].as_str().unwrap_or_default())),
                ("now", DataValue::from(now_us())),
                ("route", osv(arg_str(args, "route"))),
            ],
        )?;
        Ok(json!({"run": run, "ver": row["ver"], "name": row["name"], "scope": row["scope"]}))
    }

    /// Called when a workflow finishes: record an outcome for every skill
    /// version it used (`skill use` / a `skill` step, or `meta.skills` on
    /// start). Best effort; never fails the workflow.
    pub(crate) fn skill_flow_finished(&'s self, wf: &str, status: &str, created: Option<i64>, meta: &JsonValue) {
        if status == "canceled" || !self.sk_has(SKILL_USE) {
            return;
        }
        let mut uses: Vec<(String, Option<String>)> = self
            .sk_r(&format!("?[ver, route] := *{SKILL_USE}{{run: $r, ver, route}}"), vec![("r", sv(wf))])
            .map(|r| r.rows.iter().filter_map(|x| Some((cs(&x[0])?, cs(&x[1])))).collect())
            .unwrap_or_default();
        if let Some(a) = meta.get("skills").and_then(|s| s.as_array()) {
            for s in a {
                let args = if s.is_string() { json!({"ver": s}) } else { s.clone() };
                if let Ok(row) = self.skill_resolve(&args) {
                    uses.push((row["ver"].as_str().unwrap_or_default().to_string(), arg_str(&args, "route").map(String::from)));
                }
            }
        }
        uses.sort();
        uses.dedup_by(|a, b| a.0 == b.0);
        let latency = created.map(|c| (now_us() - c) as f64 / 1000.0);
        for (ver, route) in uses {
            let mut a = json!({"ver": ver, "run": wf, "success": status == "completed", "source": "flow"});
            if let Some(l) = latency {
                a["latency_ms"] = json!(l);
            }
            if let Some(r) = route {
                a["route"] = json!(r);
            }
            if let Err(e) = self.skill_record_outcome(&a) {
                log::warn!("skill outcome for workflow {wf}: {e}");
            }
        }
    }

    // ---------------------------------------------------------- traces, evals

    /// `record_trace {task, transcript, success, output?, run?, scope?,
    /// metrics?, skills?}`: a run transcript the learning loop harvests.
    pub(crate) fn skill_record_trace(&'s self, args: &JsonValue) -> Result<JsonValue> {
        self.skill_ensure()?;
        let task = arg_str(args, "task").ok_or_else(|| miette!("skill record_trace needs 'task'"))?;
        let transcript = args.get("transcript").map(|t| t.as_str().map(String::from).unwrap_or_else(|| t.to_string())).unwrap_or_default();
        let success = args.get("success").and_then(|v| v.as_bool()).ok_or_else(|| miette!("skill record_trace needs 'success' (bool)"))?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        let tid = arg_str(args, "id").map(String::from).unwrap_or_else(|| new_id("st"));
        let skills: Vec<String> = args.get("skills").and_then(|s| s.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        self.sk_w(
            &format!(
                "?[tid, scope, task, transcript, success, output, run, at, actor, metrics, skills] <- [[$t, $s, $task, $tr, $ok, $out, $run, $now, $a, $m, $sk]] \
                 :put {SKILL_TRACE} {{tid => scope, task, transcript, success, output, run, at, actor, metrics, skills}}"
            ),
            vec![
                ("t", sv(&tid)),
                ("s", sv(&scope)),
                ("task", sv(task)),
                ("tr", sv(&transcript)),
                ("ok", DataValue::from(success)),
                ("out", osv(arg_str(args, "output"))),
                ("run", osv(arg_str(args, "run"))),
                ("now", DataValue::from(args.get("at").and_then(|v| v.as_i64()).unwrap_or_else(now_us))),
                ("a", jdn(&actor_of(args.get("actor")))),
                ("m", args.get("metrics").map(jd).unwrap_or(DataValue::Null)),
                ("sk", str_list(&skills)),
            ],
        )?;
        Ok(json!({"id": tid, "scope": scope}))
    }

    fn skill_traces(&'s self, args: &JsonValue) -> Result<JsonValue> {
        self.skill_ensure()?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100);
        let r = self.sk_r(
            &format!("?[tid, task, success, output, run, at, skills] := *{SKILL_TRACE}{{tid, scope: $s, task, success, output, run, at, skills}} :order -at :limit {limit}"),
            vec![("s", sv(&scope))],
        )?;
        Ok(JsonValue::Array(rows_json(&r).into_iter().map(JsonValue::Object).collect()))
    }

    /// `eval_add {name, scope?, cases: [{task, expected?, check?, split?}]}`.
    pub(crate) fn skill_eval_add(&'s self, args: &JsonValue) -> Result<JsonValue> {
        self.skill_ensure()?;
        let name = arg_str(args, "name").ok_or_else(|| miette!("skill eval_add needs 'name'"))?;
        let scope = norm_scope(arg_str(args, "scope"))?;
        let cases = args.get("cases").and_then(|c| c.as_array()).ok_or_else(|| miette!("skill eval_add needs 'cases'"))?;
        let now = now_us();
        let mut rows = vec![];
        for (i, c) in cases.iter().enumerate() {
            let task = c.get("task").and_then(|t| t.as_str()).ok_or_else(|| miette!("eval case {i} needs 'task'"))?;
            let case = c.get("id").and_then(|t| t.as_str()).map(String::from).unwrap_or_else(|| format!("c{}", &sha256_hex(task.as_bytes())[..12]));
            rows.push(DataValue::List(vec![
                sv(&scope),
                sv(name),
                sv(&case),
                sv(task),
                osv(c.get("expected").and_then(|t| t.as_str())),
                c.get("check").map(jd).unwrap_or(DataValue::Null),
                sv(c.get("split").and_then(|t| t.as_str()).unwrap_or("holdout")),
                sv(c.get("source").and_then(|t| t.as_str()).unwrap_or("manual")),
                DataValue::from(now),
            ]));
        }
        let n = rows.len();
        self.sk_w(
            &format!("?[scope, name, case, task, expected, check, split, source, at] <- $rows :put {SKILL_EVAL} {{scope, name, case => task, expected, check, split, source, at}}"),
            vec![("rows", DataValue::List(rows))],
        )?;
        Ok(json!({"name": name, "scope": scope, "added": n}))
    }

    pub(crate) fn skill_evals(&'s self, scope: &str, name: &str, split: Option<&str>) -> Result<Vec<serde_json::Map<String, JsonValue>>> {
        let r = self.sk_r(
            &format!("?[case, task, expected, check, split, source] := *{SKILL_EVAL}{{scope: $s, name: $n, case, task, expected, check, split, source}}"),
            vec![("s", sv(scope)), ("n", sv(name))],
        )?;
        Ok(rows_json(&r).into_iter().filter(|c| split.map_or(true, |s| c["split"] == s)).collect())
    }

    // ---------------------------------------------------------- dispatch

    /// One entry point for every surface (`::skill <op>`, HTTP, MCP, CLI,
    /// bindings). See docs/specs/skills.md for the ops.
    pub fn skill_call(&'s self, op: &str, args: JsonValue) -> Result<JsonValue> {
        let a = &args;
        if !matches!(op, "init" | "keygen") {
            self.skill_ensure()?;
        }
        Ok(match op {
            "init" => self.skill_init(a)?,
            "config" => self.skill_config(a)?,
            "put" => self.skill_put(a)?,
            "import" => self.skill_import(a)?,
            "export" => self.skill_export(a)?,
            "get" => self.skill_get(a)?,
            "list" => self.skill_list(a)?,
            "history" => self.skill_history(a)?,
            "as_of" => self.skill_as_of(a)?,
            "find" => self.skill_find(a)?,
            "entry" => self.skill_entry_op(a)?,
            "entries" => self.skill_entries(a)?,
            "read" => self.skill_read_uri(a)?,
            "dir" => self.skill_dir_uri(a)?,
            "activate" => self.skill_activate(a)?,
            "rollback" => self.skill_rollback(a)?,
            "undo" => self.skill_undo(a)?,
            "retire" => self.skill_retire(a)?,
            "quarantine" => self.skill_quarantine(a)?,
            "release" => self.skill_release(a)?,
            "erase" => self.skill_erase(a)?,
            "lifecycle" => self.skill_lifecycle(a)?,
            "scan" => self.skill_scan_op(a)?,
            "record_outcome" | "outcome" => self.skill_record_outcome(a)?,
            "use" => self.skill_use(a)?,
            "stats" => self.skill_stats(a)?,
            "record_trace" | "trace" => self.skill_record_trace(a)?,
            "traces" => self.skill_traces(a)?,
            "eval_add" => self.skill_eval_add(a)?,
            "evals" => {
                let name = arg_str(a, "name").ok_or_else(|| miette!("skill evals needs 'name'"))?;
                let scope = norm_scope(arg_str(a, "scope"))?;
                JsonValue::Array(self.skill_evals(&scope, name, arg_str(a, "split"))?.into_iter().map(JsonValue::Object).collect())
            }
            "eval" => self.skill_eval_op(a)?,
            "learn" => self.skill_learn(a)?,
            "learn_stage" => self.skill_learn_stage(a)?,
            "optimize" => self.skill_optimize(a)?,
            "optimize_stage" => self.skill_optimize_stage(a)?,
            "proposals" => self.skill_proposals(a)?,
            "propose" => self.skill_propose(a)?,
            "promote" => self.skill_promote(a)?,
            "approve" => self.skill_approve(a)?,
            "reject" => self.skill_reject(a)?,
            "promotions" => self.skill_promotions(a)?,
            "keygen" => crate::skill_sign::keygen(arg_str(a, "label"))?,
            "trust" => self.skill_trust(a)?,
            "untrust" => self.skill_untrust(a)?,
            "keys" => self.skill_keys()?,
            "sign" => self.skill_sign(a)?,
            "verify" => self.skill_verify(a)?,
            "package" => self.skill_package(a)?,
            "publish" => self.skill_publish(a)?,
            "registry" => self.skill_registry(a)?,
            "install" => self.skill_install(a)?,
            other => bail!(
                "unknown skill op '{other}': expected init, config, put, import, export, get, list, history, as_of, find, entry, entries, \
                 read, dir, activate, rollback, undo, retire, quarantine, release, erase, lifecycle, scan, record_outcome, use, stats, \
                 record_trace, traces, eval_add, evals, eval, learn, optimize, proposals, propose, promote, approve, reject, promotions, \
                 keygen, trust, untrust, keys, sign, verify, package, publish, registry or install"
            ),
        })
    }
}

/// Aggregated outcomes of one version.
#[derive(Debug, Clone, Default)]
pub(crate) struct OutcomeAgg {
    pub uses: i64,
    pub successes: i64,
    pub lat_sum: f64,
    pub lat_n: i64,
    pub tok_sum: i64,
    pub tok_n: i64,
    pub cost: f64,
    pub last: i64,
}

impl OutcomeAgg {
    /// Posterior mean of the success rate (uniform prior).
    pub fn posterior(&self) -> f64 {
        (self.successes as f64 + 1.0) / (self.uses as f64 + 2.0)
    }
    pub fn to_json(&self) -> JsonValue {
        json!({
            "uses": self.uses,
            "successes": self.successes,
            "failures": self.uses - self.successes,
            "success_rate": if self.uses > 0 { json!(self.successes as f64 / self.uses as f64) } else { JsonValue::Null },
            "posterior": self.posterior(),
            "latency_ms": if self.lat_n > 0 { json!(self.lat_sum / self.lat_n as f64) } else { JsonValue::Null },
            "tokens": if self.tok_n > 0 { json!(self.tok_sum as f64 / self.tok_n as f64) } else { JsonValue::Null },
            "cost": self.cost,
            "last_used": if self.last > 0 { json!(self.last) } else { JsonValue::Null },
        })
    }
}

fn norm_cdf(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26 on erf
    let t = 1.0 / (1.0 + 0.3275911 * (x.abs() / std::f64::consts::SQRT_2));
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-(x * x) / 2.0).exp();
    if x >= 0.0 {
        0.5 * (1.0 + y)
    } else {
        0.5 * (1.0 - y)
    }
}

fn beta_moments(s: i64, f: i64) -> (f64, f64) {
    let (a, b) = (s as f64 + 1.0, f as f64 + 1.0);
    let m = a / (a + b);
    (m, a * b / ((a + b).powi(2) * (a + b + 1.0)))
}

/// P(p_a < p_b) for Beta(1+s, 1+f) posteriors (normal approximation).
pub fn prob_less(sa: i64, fa: i64, sb: i64, fb: i64) -> f64 {
    let (ma, va) = beta_moments(sa, fa);
    let (mb, vb) = beta_moments(sb, fb);
    norm_cdf((mb - ma) / (va + vb).sqrt().max(1e-9))
}

/// P(p < x) for a Beta(1+s, 1+f) posterior (normal approximation).
pub fn prob_below(s: i64, f: i64, x: f64) -> f64 {
    let (m, v) = beta_moments(s, f);
    norm_cdf((x - m) / v.sqrt().max(1e-9))
}

/// Deep-merge `patch` into `base` (objects merge, everything else replaces).
pub(crate) fn merge(base: &mut JsonValue, patch: &JsonValue) {
    match (base, patch) {
        (JsonValue::Object(b), JsonValue::Object(p)) => {
            for (k, v) in p {
                merge(b.entry(k.clone()).or_insert(JsonValue::Null), v);
            }
        }
        (b, p) => *b = p.clone(),
    }
}

/// `SKILL.md` with `kleos.*` keys added to the frontmatter `metadata` (the
/// rest of the file unchanged).
fn annotate_md(md: &[u8], row: &serde_json::Map<String, JsonValue>, stats: &JsonValue) -> Result<Vec<u8>> {
    let text = String::from_utf8_lossy(md).into_owned();
    let doc = parse_skill_md(&text)?;
    let mut y: serde_yaml::Mapping = serde_yaml::from_str(&doc.fm_text).into_diagnostic()?;
    let key = serde_yaml::Value::from("metadata");
    let mut meta = match y.get(&key) {
        Some(serde_yaml::Value::Mapping(m)) => m.clone(),
        _ => serde_yaml::Mapping::new(),
    };
    let mut put = |k: &str, v: String| {
        meta.insert(serde_yaml::Value::from(k), serde_yaml::Value::from(v));
    };
    put("kleos.version", row.get("seq").map(|v| v.to_string()).unwrap_or_default());
    put("kleos.digest", row.get("digest").and_then(|v| v.as_str()).unwrap_or_default().to_string());
    put("kleos.scope", row.get("scope").and_then(|v| v.as_str()).unwrap_or_default().to_string());
    if let Some(r) = stats.get("success_rate").and_then(|v| v.as_f64()) {
        put("kleos.success_rate", format!("{r:.3}"));
        put("kleos.uses", stats["uses"].to_string());
    }
    y.insert(key, serde_yaml::Value::Mapping(meta));
    let fm = serde_yaml::to_string(&y).into_diagnostic()?;
    let fm = fm.strip_prefix("---\n").unwrap_or(&fm).to_string();
    Ok(format!("---\n{fm}---\n{}", doc.body).into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frontmatter_verbatim() {
        let md = "---\nname: pdf-tools\ndescription: Fill PDF forms.\nlicense: MIT\nmetadata:\n  version: \"1.0\"\n---\n# Body\n\nText.\n";
        let d = parse_skill_md(md).unwrap();
        assert_eq!(d.name, "pdf-tools");
        assert_eq!(d.frontmatter["metadata"]["version"], "1.0");
        assert_eq!(d.frontmatter["license"], "MIT");
        assert!(d.body.starts_with("# Body"));
        assert_eq!(with_body(&d, &d.body), md);
        assert!(parse_skill_md("---\nname: Bad_Name\ndescription: x\n---\n").is_err());
        assert!(parse_skill_md("no frontmatter").is_err());
    }

    #[test]
    fn uris_round_trip() {
        assert_eq!(skill_uri("default", "a-b", "SKILL.md"), "skill://a-b/SKILL.md");
        assert_eq!(skill_uri("team:billing", "refunds", "SKILL.md"), "skill://team/billing/refunds/SKILL.md");
        let s = uri_splits("skill://team/billing/refunds/references/x.md");
        assert!(s.contains(&("team:billing".into(), "refunds".into(), "references/x.md".into())));
        assert_eq!(scope_rank("agent:x"), 3);
        assert!(scope_rank("org") > scope_rank("ws"));
    }

    #[test]
    fn bayes_comparisons() {
        assert!(prob_less(5, 25, 25, 5) > 0.99);
        assert!(prob_less(15, 15, 15, 15) > 0.4 && prob_less(15, 15, 15, 15) < 0.6);
        assert!(prob_below(2, 28, 0.25) > 0.95);
    }
}
