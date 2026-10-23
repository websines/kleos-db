/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The model-free parts of `recall` (docs/specs/memory-api.md, "Diversity"
//! and "Context output"): diverse selection from the candidate pool (caps per
//! statement and per source group, MMR-style near-duplicate penalty) and the
//! token-budgeted text block for an LLM reader. Everything here is
//! deterministic: ties break on the candidate's rank, then its id.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{TimeZone, Utc};
use serde_derive::{Deserialize, Serialize};

/// The statement a memory came from: a claim `m_….n` belongs to `m_…`; any
/// other id is its own statement.
pub fn statement_of(id: &str) -> &str {
    match id.rsplit_once('.') {
        Some((stmt, n)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => stmt,
        _ => id,
    }
}

/// The diversity group of a source: the part before the first `#`
/// (`chat-12#4` and `chat-12#9` are one conversation), else the whole source.
pub fn source_group(source: &str) -> &str {
    source.split_once('#').map_or(source, |(g, _)| g)
}

/// Lower-case words of at least three letters or digits.
pub fn word_set(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() > 2)
        .map(|w| w.to_lowercase())
        .collect()
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f64;
    inter / (a.len() as f64 + b.len() as f64 - inter)
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

/// One candidate of the diverse selection, in rank (score) order.
pub struct SelCand<'a> {
    pub id: &'a str,
    pub score: f64,
    /// Source group (see [`source_group`]); `None`: never capped by source.
    pub group: Option<&'a str>,
    pub vec: Option<&'a [f32]>,
    pub words: BTreeSet<String>,
}

/// Similarity of two candidates in [0, 1]: the cosine of their vectors when
/// both have one, else the Jaccard overlap of their words.
pub fn similarity(a: &SelCand<'_>, b: &SelCand<'_>) -> f64 {
    match (a.vec, b.vec) {
        (Some(x), Some(y)) => cos(x, y).max(0.0),
        _ => jaccard(&a.words, &b.words),
    }
}

/// Diverse top-k from `cands` (sorted by score, best first): greedily take
/// the candidate with the highest `score − mmr_weight · max similarity to
/// the ones already taken` among those within the caps (`max_per_statement`
/// hits per statement, `max_per_source` per source group; 0 = no cap). When
/// the caps leave fewer than `k`, the capped candidates fill the rest in
/// score order. Returns indexes into `cands`, in selection order.
pub fn select_diverse(
    cands: &[SelCand<'_>],
    k: usize,
    max_per_statement: usize,
    max_per_source: usize,
    mmr_weight: f64,
) -> Vec<usize> {
    let n = cands.len();
    let mut taken = vec![false; n];
    let mut maxsim = vec![0f64; n];
    let mut per_stmt: BTreeMap<&str, usize> = BTreeMap::new();
    let mut per_group: BTreeMap<&str, usize> = BTreeMap::new();
    let mut out = Vec::with_capacity(k.min(n));
    while out.len() < k {
        let mut best: Option<(usize, f64)> = None;
        for i in 0..n {
            if taken[i] {
                continue;
            }
            let c = &cands[i];
            if max_per_statement > 0 && per_stmt.get(statement_of(c.id)).copied().unwrap_or(0) >= max_per_statement {
                continue;
            }
            if max_per_source > 0 {
                if let Some(g) = c.group {
                    if per_group.get(g).copied().unwrap_or(0) >= max_per_source {
                        continue;
                    }
                }
            }
            let v = c.score - mmr_weight * maxsim[i];
            // strictly greater: ties keep the earlier (higher-ranked) one
            if best.map_or(true, |(_, b)| v > b) {
                best = Some((i, v));
            }
        }
        let Some((i, _)) = best else { break };
        taken[i] = true;
        out.push(i);
        *per_stmt.entry(statement_of(cands[i].id)).or_default() += 1;
        if let Some(g) = cands[i].group {
            *per_group.entry(g).or_default() += 1;
        }
        if mmr_weight > 0.0 {
            for j in 0..n {
                if !taken[j] {
                    let s = similarity(&cands[i], &cands[j]);
                    if s > maxsim[j] {
                        maxsim[j] = s;
                    }
                }
            }
        }
    }
    // the caps left too few: fill in score order
    for i in 0..n {
        if out.len() >= k {
            break;
        }
        if !taken[i] {
            taken[i] = true;
            out.push(i);
        }
    }
    out
}

// ------------------------------------------------------------------ context output

/// What `recall {format: 'context'}` returns: a text block for an LLM reader.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RecallContext {
    /// The block (Markdown).
    pub text: String,
    /// Estimated tokens of `text` (characters / 4, rounded up).
    pub tokens: usize,
    /// The token budget it was built for.
    pub budget: usize,
    /// Memories shown, in rank order.
    pub ids: Vec<String>,
    /// Recalled memories left out because the budget was spent.
    pub omitted: Vec<String>,
}

/// One earlier value of a memory: the memory it superseded.
#[derive(Debug, Clone, PartialEq)]
pub struct Previous {
    pub text: String,
    pub from: i64,
    pub to: Option<i64>,
}

/// One recalled memory as the context block needs it, in rank order.
#[derive(Debug, Clone, PartialEq)]
pub struct CtxItem {
    pub id: String,
    pub text: String,
    pub valid_from: i64,
    pub valid_to: Option<i64>,
    pub role: Option<String>,
    pub source: Option<String>,
    /// The whole statement, when the memory is one claim of it.
    pub context: Option<String>,
    /// Canonical entity names the memory mentions.
    pub entities: Vec<String>,
    /// The values it replaced, newest first.
    pub previously: Vec<Previous>,
    /// Texts of newer memories that may replace it (pending decisions).
    pub maybe_replaced_by: Vec<(String, i64)>,
}

/// Estimated tokens: characters / 4, rounded up (no tokenizer, so the
/// estimate is the same everywhere).
pub fn estimate_tokens(s: &str) -> usize {
    s.chars().count().div_ceil(4)
}

pub fn fmt_date(us: i64) -> String {
    match Utc.timestamp_micros(us).single() {
        Some(d) => d.format("%Y-%m-%d (%a)").to_string(),
        None => us.to_string(),
    }
}

fn day(us: i64) -> String {
    match Utc.timestamp_micros(us).single() {
        Some(d) => d.format("%Y-%m-%d").to_string(),
        None => us.to_string(),
    }
}

/// One line, whitespace collapsed, quotes kept.
fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// At most `max` characters of `context` around `claim` (or its start),
/// cut at word boundaries, with `…` where cut.
pub fn window(context: &str, claim: &str, max: usize) -> String {
    let ctx = one_line(context);
    let chars: Vec<char> = ctx.chars().collect();
    if chars.len() <= max {
        return ctx;
    }
    let cl = one_line(claim);
    let pos = ctx.find(&cl).map(|b| ctx[..b].chars().count()).unwrap_or(0);
    let cl_len = cl.chars().count();
    // centre the window on the claim
    let slack = max.saturating_sub(cl_len.min(max));
    let mut start = pos.saturating_sub(slack / 2);
    if start + max > chars.len() {
        start = chars.len() - max;
    }
    let mut end = (start + max).min(chars.len());
    // word boundaries
    while start > 0 && start < chars.len() && !chars[start - 1].is_whitespace() {
        start += 1;
        if start >= end {
            break;
        }
    }
    while end < chars.len() && end > start && !chars[end].is_whitespace() {
        end -= 1;
    }
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.extend(chars[start..end].iter());
    let out = out.trim_end().to_string();
    if end < chars.len() {
        out + "…"
    } else {
        out
    }
}

/// Characters of statement context shown per item.
pub const CONTEXT_CHARS: usize = 600;

/// The header line of every context block.
pub const CONTEXT_HEADER: &str = "Recalled memories, grouped by topic, oldest first. [date] is when a statement was made or became true; \"previously\" lines are older values it replaced.";

/// Render the context block from `items` (rank order) within `budget`
/// tokens. Items are grouped by entity (the canonical entity most of the
/// shown memories mention; ties by name), else by source group, else
/// "Other"; groups are ordered by their first item's date, items in a group
/// by date. Claims of one statement are shown as one item. First every item
/// that fits is added in rank order, with its earlier values; then the
/// statement contexts, in rank order, while the budget lasts.
pub fn render_context(items: &[CtxItem], budget: usize, show_source: bool) -> RecallContext {
    // merge claims of one statement (rank of the best claim)
    let mut merged: Vec<CtxItem> = vec![];
    let mut ids_of: Vec<Vec<String>> = vec![];
    let mut at: BTreeMap<String, usize> = BTreeMap::new();
    for it in items {
        let st = statement_of(&it.id).to_string();
        if let Some(&i) = at.get(&st) {
            let m = &mut merged[i];
            if !m.text.contains(it.text.trim()) {
                m.text = format!("{} {}", m.text.trim_end(), it.text.trim());
            }
            for e in &it.entities {
                if !m.entities.contains(e) {
                    m.entities.push(e.clone());
                }
            }
            m.previously.extend(it.previously.iter().cloned());
            m.maybe_replaced_by.extend(it.maybe_replaced_by.iter().cloned());
            m.valid_from = m.valid_from.min(it.valid_from);
            ids_of[i].push(it.id.clone());
        } else {
            at.insert(st, merged.len());
            merged.push(it.clone());
            ids_of.push(vec![it.id.clone()]);
        }
    }
    // the group of each item
    let mut ent_count: BTreeMap<&str, usize> = BTreeMap::new();
    for it in &merged {
        for e in &it.entities {
            *ent_count.entry(e.as_str()).or_default() += 1;
        }
    }
    let group_of: Vec<String> = merged
        .iter()
        .map(|it| {
            let best = it
                .entities
                .iter()
                .max_by(|a, b| {
                    ent_count[a.as_str()]
                        .cmp(&ent_count[b.as_str()])
                        .then_with(|| b.cmp(a))
                })
                .cloned();
            match (best, &it.source) {
                (Some(e), _) => format!("e:{e}"),
                (None, Some(s)) => format!("s:{}", source_group(s)),
                (None, None) => "o:".to_string(),
            }
        })
        .collect();

    // pass 1: items in rank order while they fit; pass 2: contexts
    let header_tokens = estimate_tokens(CONTEXT_HEADER) + 1;
    let mut used = header_tokens;
    let mut shown = vec![false; merged.len()];
    let mut lines_of: Vec<Vec<String>> = vec![vec![]; merged.len()];
    let mut heads: BTreeSet<&str> = BTreeSet::new();
    let head_cost = 12; // a group heading, estimated
    for (i, it) in merged.iter().enumerate() {
        let mut lines = vec![];
        let mut first = format!("- [{}]", fmt_date(it.valid_from));
        if let Some(to) = it.valid_to {
            first.push_str(&format!(" (until {})", day(to)));
        }
        if let Some(r) = it.role.as_deref().filter(|r| *r != "user") {
            first.push_str(&format!(" ({r})"));
        }
        first.push(' ');
        first.push_str(&one_line(&it.text));
        if show_source {
            if let Some(s) = &it.source {
                first.push_str(&format!(" [source: {s}]"));
            }
        }
        lines.push(first);
        let mut seen_prev = BTreeSet::new();
        for p in it.previously.iter() {
            if !seen_prev.insert(p.text.clone()) || seen_prev.len() > 3 {
                continue;
            }
            let until = p.to.map(|t| format!(" until {}", day(t))).unwrap_or_default();
            lines.push(format!("  previously: \"{}\" (from {}{until})", one_line(&p.text), day(p.from)));
        }
        for (t, from) in it.maybe_replaced_by.iter().take(2) {
            lines.push(format!("  possibly replaced by: \"{}\" ({}, unconfirmed)", one_line(t), day(*from)));
        }
        let cost: usize = lines.iter().map(|l| estimate_tokens(l) + 1).sum::<usize>()
            + if heads.contains(group_of[i].as_str()) { 0 } else { head_cost };
        if used + cost > budget {
            continue;
        }
        used += cost;
        heads.insert(group_of[i].as_str());
        shown[i] = true;
        lines_of[i] = lines;
    }
    for (i, it) in merged.iter().enumerate() {
        if !shown[i] {
            continue;
        }
        let Some(c) = it.context.as_deref() else { continue };
        let w = window(c, &it.text, CONTEXT_CHARS);
        if one_line(&w) == one_line(&it.text) {
            continue;
        }
        let line = format!("  said in: \"{w}\"");
        let cost = estimate_tokens(&line) + 1;
        if used + cost > budget {
            continue;
        }
        used += cost;
        lines_of[i].push(line);
    }

    // assemble: groups in time order (by their first item; ties by the
    // best rank), items in a group by date, so the newest value reads last
    // as in a dated list
    let mut order: Vec<&str> = vec![];
    for (i, g) in group_of.iter().enumerate() {
        if shown[i] && !order.contains(&g.as_str()) {
            order.push(g.as_str());
        }
    }
    let first_at = |g: &str| {
        (0..merged.len())
            .filter(|&i| shown[i] && group_of[i] == g)
            .map(|i| merged[i].valid_from)
            .min()
            .unwrap_or(i64::MAX)
    };
    let mut keyed: Vec<(i64, usize, &str)> = order.iter().enumerate().map(|(r, g)| (first_at(g), r, *g)).collect();
    keyed.sort();
    let order: Vec<&str> = keyed.into_iter().map(|x| x.2).collect();
    let mut text = String::from(CONTEXT_HEADER);
    text.push('\n');
    let mut ids = vec![];
    for g in order {
        let mut members: Vec<usize> = (0..merged.len()).filter(|&i| shown[i] && group_of[i] == g).collect();
        members.sort_by(|&a, &b| merged[a].valid_from.cmp(&merged[b].valid_from).then(a.cmp(&b)));
        let label = match g.split_once(':') {
            Some(("e", name)) => name.to_string(),
            Some(("s", _)) => {
                let lo = members.iter().map(|&i| merged[i].valid_from).min().unwrap_or(0);
                let hi = members.iter().map(|&i| merged[i].valid_from).max().unwrap_or(0);
                if day(lo) == day(hi) {
                    format!("Conversation of {}", day(lo))
                } else {
                    format!("Conversation of {} to {}", day(lo), day(hi))
                }
            }
            _ => "Other".to_string(),
        };
        text.push_str(&format!("\n## {label}\n"));
        for i in members {
            for l in &lines_of[i] {
                text.push_str(l);
                text.push('\n');
            }
        }
    }
    for (i, idl) in ids_of.iter().enumerate() {
        if shown[i] {
            ids.extend(idl.iter().cloned());
        }
    }
    // ids in rank order
    let rank: BTreeMap<&str, usize> = items.iter().enumerate().map(|(i, it)| (it.id.as_str(), i)).collect();
    ids.sort_by_key(|id| rank.get(id.as_str()).copied().unwrap_or(usize::MAX));
    let shown_set: BTreeSet<&str> = ids.iter().map(|s| s.as_str()).collect();
    let omitted = items
        .iter()
        .filter(|it| !shown_set.contains(it.id.as_str()))
        .map(|it| it.id.clone())
        .collect();
    let tokens = estimate_tokens(&text);
    RecallContext {
        text,
        tokens,
        budget,
        ids,
        omitted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statements_and_groups() {
        assert_eq!(statement_of("m_00ab.3"), "m_00ab");
        assert_eq!(statement_of("m_00ab"), "m_00ab");
        assert_eq!(statement_of("doc.v2"), "doc.v2");
        assert_eq!(source_group("chat-12#4"), "chat-12");
        assert_eq!(source_group("chat-12"), "chat-12");
    }

    #[test]
    fn windows_cut_at_words() {
        let ctx = "one two three four five six seven eight nine ten eleven twelve";
        let w = window(ctx, "six seven", 20);
        assert!(w.contains("six seven"), "{w}");
        assert!(w.starts_with('…') && w.ends_with('…'), "{w}");
        assert_eq!(window("short text", "short", 100), "short text");
    }
}
