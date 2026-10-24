/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Graph-aware recall (docs/specs/memory-api.md, "Graph recall"):
//!
//! - `recall {graph: 'inferred'}` pulls in memories linked to the query and
//!   the top hits through the ontology's facts, inferred ones included
//!   (symmetric, inverse, sub-property, transitive, chains, `same_as`):
//!   "who are my relatives?" reaches the sister and brother facts through
//!   `relative_of`. Each such hit says why (`via: "inferred: relative_of"`).
//! - `recall {graph: 'ppr'}` re-ranks with a personalized PageRank over the
//!   memory / entity / ontology-node graph around the hits (HippoRAG 2,
//!   arXiv 2502.14802): the hybrid hits and the nodes the query names are
//!   the restart distribution, mentions, claim subjects and ontology facts
//!   (inferred included) are the edges. It also does the `inferred`
//!   expansion.
//!
//! Both are bounded: the graph is the neighbourhood of the seeds (two hops,
//! high-degree entities not expanded, at most `PPR_MAX_MEMS` memories), so
//! the cost does not grow with the store.

use super::*;

/// Recall graph modes.
pub(crate) const GRAPH_MODES: &[&str] = &["off", "inferred", "ppr"];
/// Default graph mode: `ppr`, which also runs the inferred-link expansion
/// (the user's choice; trade-offs in docs/benchmarks/memory.md). `graph:
/// 'off'` opts out.
pub(crate) const DEFAULT_GRAPH_MODE: &str = "ppr";
/// Share of the PPR score in a `ppr` re-rank.
pub(crate) const DEFAULT_GRAPH_WEIGHT: f64 = 0.3;
/// Restart probability complement (HippoRAG 2 uses 0.5).
const PPR_DAMPING: f64 = 0.5;
const PPR_ITERS: usize = 30;
/// Hits that seed the graph.
const GRAPH_SEEDS: usize = 10;
/// At most this many memories in a PPR neighbourhood.
const PPR_MAX_MEMS: usize = 2000;
/// New memories a graph mode may add to the pool.
const GRAPH_MAX_NEW: usize = 20;
/// An entity or node with more memories than this is a hub: cut from the
/// PageRank graph.
const HUB_DEGREE: usize = 60;
/// Hits whose subjects and facts seed the inferred expansion.
const INFER_FROM: usize = 5;

/// What a graph mode changes in the search rows.
#[derive(Default)]
pub(crate) struct GraphOut {
    /// New relevance of rows already in the pool (`ppr` replaces, `inferred`
    /// raises).
    pub set: BTreeMap<String, f64>,
    pub replace: bool,
    /// Rows for memories not in the pool (search-row layout).
    pub new_rows: Vec<Vec<DataValue>>,
    /// Why a memory was included.
    pub via: BTreeMap<String, String>,
}

/// Node keys are lower case, single-spaced (as the ontology's kinship rules
/// write them).
fn norm(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

fn node_of_subject(s: &str) -> String {
    norm(s.strip_prefix("name:").unwrap_or(s))
}

fn singular(w: &str) -> String {
    if w.len() > 4 && w.ends_with("ies") {
        format!("{}y", &w[..w.len() - 3])
    } else if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") {
        w[..w.len() - 1].to_string()
    } else {
        w.to_string()
    }
}

const FIRST_PERSON: &[&str] = &["i", "my", "me", "mine", "myself", "we", "our", "us", "ours"];
const STOP: &[&str] = &["of", "in", "the", "a", "an", "to", "is", "has", "and", "or", "for", "on", "at", "by", "as"];

/// Properties the query asks about: an alias phrase in the query, or a
/// content word of the property's name equal to a query word (singular).
pub(crate) fn query_props(act: &crate::ontology::Active, query: &str) -> BTreeSet<String> {
    let low = format!(" {} ", query.to_lowercase().replace(|c: char| !c.is_alphanumeric() && c != '\'', " "));
    let words: BTreeSet<String> = low.split_whitespace().map(singular).collect();
    let mut out = BTreeSet::new();
    for p in &act.props {
        if p.name == "same_as" {
            continue;
        }
        let alias_hit = p.aliases.iter().any(|a| {
            let a = a.to_lowercase();
            !a.is_empty() && (low.contains(&format!(" {a} ")) || words.contains(&singular(&a)))
        });
        let name_hit = p
            .name
            .split('_')
            .filter(|w| w.len() > 2 && !STOP.contains(w))
            .any(|w| words.contains(&singular(w)));
        if alias_hit || name_hit {
            out.insert(p.name.clone());
        }
    }
    out
}

/// Nodes the query names: `user` for first-person words, `user.<kin>` for
/// "my sister", and any node key that appears as a word.
fn query_nodes(query: &str, nodes: &BTreeSet<String>) -> BTreeSet<String> {
    let toks: Vec<String> = query
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    let mut out = BTreeSet::new();
    if toks.iter().any(|t| FIRST_PERSON.contains(&t.as_str())) && nodes.contains("user") {
        out.insert("user".to_string());
    }
    for (i, t) in toks.iter().enumerate() {
        if (t == "my" || t == "our") && i + 1 < toks.len() {
            let k = format!("user.{}", singular(toks[i + 1].trim_end_matches("'s")));
            if nodes.contains(&k) {
                out.insert(k);
            }
        }
        let w = t.trim_end_matches("'s");
        if w.len() > 1 && !FIRST_PERSON.contains(&w) && nodes.contains(w) {
            out.insert(w.to_string());
        }
    }
    // two-word names ("anna maria")
    for w in toks.windows(2) {
        let k = format!("{} {}", w[0], w[1].trim_end_matches("'s"));
        if nodes.contains(&k) {
            out.insert(k);
        }
    }
    out
}

/// The ontology's object facts of a namespace (its lineage read at each
/// level's cut), with the memories that stated them, and everything the
/// active ontology infers from them.
pub(crate) struct OntoGraph {
    /// (s, p, o, derived)
    pub facts: Vec<(String, String, String, bool)>,
    /// base (s, p, o) → evidence memories
    pub evidence: BTreeMap<(String, String, String), BTreeSet<String>>,
    pub nodes: BTreeSet<String>,
}

impl<'s, S: Storage<'s>> Db<S> {
    fn rel_exists(&'s self, name: &str) -> bool {
        self.transact().map(|tx| tx.get_relation(name, false).is_ok()).unwrap_or(false)
    }

    /// The namespace's ontology graph, or `None` without an active ontology
    /// or facts.
    pub(crate) fn onto_graph(&'s self, ns: &str) -> Result<Option<OntoGraph>> {
        if !self.rel_exists(crate::ontology::ONTO_FACT) {
            return Ok(None);
        }
        let Some(_act) = self.onto_active(ns)? else { return Ok(None) };
        let mut evidence: BTreeMap<(String, String, String), BTreeSet<String>> = BTreeMap::new();
        let mut nodes = BTreeSet::new();
        for (l, cut) in self.onto_levels(ns, None) {
            let at = cut.map_or_else(|| "'NOW'".to_string(), |t| t.to_string());
            for r in self
                .mem_q(
                    &format!(
                        "?[s, p, o, mem] := *{}{{ns: $ns, s, p, o, lit, mem, status @ {at}}}, status = 'current', is_null(lit)",
                        crate::ontology::ONTO_FACT
                    ),
                    vec![("ns", s(&l))],
                )?
                .rows
            {
                let (Some(fs), Some(fp), Some(fo)) = (cell_str(&r[0]), cell_str(&r[1]), cell_str(&r[2])) else { continue };
                nodes.insert(fs.clone());
                nodes.insert(fo.clone());
                let e = evidence.entry((fs, fp, fo)).or_default();
                if let Some(m) = cell_str(&r[3]) {
                    e.insert(m);
                }
            }
        }
        if evidence.is_empty() {
            return Ok(None);
        }
        let facts: Vec<(String, String, String, bool)> = self
            .onto_infer(ns, None, &None)?
            .into_iter()
            .filter(|(fs, _, fo, _)| nodes.contains(fs) && nodes.contains(fo))
            .collect();
        Ok(Some(OntoGraph { facts, evidence, nodes }))
    }

    /// Claim subjects (`kleos_mem_type`) of `ids`, as node keys.
    pub(crate) fn subjects_of(&'s self, sc: &Scope, ids: &[String]) -> Result<Vec<(String, String)>> {
        if ids.is_empty() || !self.rel_exists(crate::ontology::MEM_TYPE) {
            return Ok(vec![]);
        }
        let mut out = vec![];
        for (l, group) in self.by_owner(sc, ids)? {
            for r in self
                .mem_q(
                    &format!("?[id, subject] := id in $ids, *{}{{ns: $ns, id, subject}}, !is_null(subject)", crate::ontology::MEM_TYPE),
                    vec![("ns", s(&sc.levels[l].ns)), ("ids", str_list(&group))],
                )?
                .rows
            {
                if let (Some(id), Some(sub)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    out.push((id, node_of_subject(&sub)));
                }
            }
        }
        Ok(out)
    }

    /// Memories whose claim subject is one of `nodes`.
    fn about_nodes(&'s self, sc: &Scope, nodes: &BTreeSet<String>) -> Result<Vec<(String, String)>> {
        if nodes.is_empty() || !self.rel_exists(crate::ontology::MEM_TYPE) {
            return Ok(vec![]);
        }
        // subjects are stored as written: a bare key or "name:<key>"
        let mut keys: Vec<String> = nodes.iter().cloned().collect();
        keys.extend(nodes.iter().map(|n| format!("name:{n}")));
        let mut out = BTreeSet::new();
        for l in &sc.levels {
            for r in self
                .mem_q(
                    &format!("?[id, subject] := *{}{{ns: $ns, id, subject}}, is_in(subject, $keys)", crate::ontology::MEM_TYPE),
                    vec![("ns", s(&l.ns)), ("keys", str_list(&keys))],
                )?
                .rows
            {
                if let (Some(id), Some(sub)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    out.insert((id, node_of_subject(&sub)));
                }
            }
        }
        Ok(out.into_iter().collect())
    }

    /// Search rows for `cand` (id, score) valid at the read time, each from
    /// its owner (the layout of `mem_search`).
    pub(crate) fn rows_of(
        &'s self,
        scope: &Scope,
        cand: &[(String, f64)],
        as_of: Option<i64>,
        kind: Option<&str>,
    ) -> Result<Vec<Vec<DataValue>>> {
        let ids: Vec<String> = cand.iter().map(|x| x.0.clone()).collect();
        let mut out = vec![];
        for (l, group) in self.by_owner(scope, &ids)? {
            let mut p = vec![
                ("ns", s(&scope.levels[l].ns)),
                (
                    "cands",
                    DataValue::List(
                        cand.iter()
                            .filter(|(id, _)| group.contains(id))
                            .map(|(id, sc)| DataValue::List(vec![s(id), DataValue::from(*sc)]))
                            .collect(),
                    ),
                ),
            ];
            let at = match scope.level_at(l, as_of) {
                Some(t) => {
                    p.push(("as_of", DataValue::from(t)));
                    "$as_of"
                }
                None => "'NOW'",
            };
            let mut filters = String::new();
            if let Some(kd) = kind {
                match kd.strip_prefix('!') {
                    Some(not) => {
                        filters.push_str(", kind != $kind");
                        p.push(("kind", s(not)));
                    }
                    None => {
                        filters.push_str(", kind == $kind");
                        p.push(("kind", s(kd)));
                    }
                }
            }
            out.extend(
                self.mem_q(
                    &format!(
                        "c[id, score] <- $cands\n\
                         ?[id, score, text, kind, source, created, importance] := c[id, score], \
                           *{MEM}{{ns: $ns, id, text, kind, source, created, importance @ {at}}}{filters}"
                    ),
                    p,
                )?
                .rows,
            );
        }
        out.sort_by(|a, b| {
            b[1].get_float()
                .partial_cmp(&a[1].get_float())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| cell_str(&a[0]).cmp(&cell_str(&b[0])))
        });
        Ok(out)
    }

    /// The entity view and normalised entity name → canonical entity ids.
    fn entity_names(&'s self, sc: &Scope) -> Result<(EntityView, BTreeMap<String, BTreeSet<String>>)> {
        let view = self.entity_view(sc)?;
        let mut by_name: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (e, (name, _)) in &view.rows {
            by_name.entry(norm(name)).or_default().insert(view.canonical(e));
        }
        Ok((view, by_name))
    }

    /// The `graph` recall modes over the search `rows` (see the module
    /// docs). `weight` is the expansion weight, `gw` the PPR share.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn graph_recall(
        &'s self,
        sc: &Scope,
        ns: &str,
        query: &str,
        mode: &str,
        rows: &[Vec<DataValue>],
        weight: f64,
        gw: f64,
        as_of: Option<i64>,
        kind: Option<&str>,
    ) -> Result<GraphOut> {
        let og = self.onto_graph(ns)?;
        let act = self.onto_active(ns)?;
        let mut out = match mode {
            "ppr" => self.ppr_rerank(sc, query, rows, og.as_ref(), gw, as_of, kind)?,
            _ => GraphOut::default(),
        };
        if let (Some(og), Some(act)) = (og.as_ref(), act.as_ref()) {
            let inf = self.inferred_expand(sc, query, rows, og, act, weight, as_of, kind)?;
            let have: BTreeSet<String> = rows.iter().filter_map(|r| cell_str(&r[0])).collect();
            let added: BTreeSet<String> = out.new_rows.iter().filter_map(|r| cell_str(&r[0])).collect();
            for (id, v) in inf.set {
                // `ppr` replaces relevance: an inferred link can only raise
                // its result; alone, the caller raises relevance
                let e = out.set.entry(id).or_insert(v);
                if v > *e {
                    *e = v;
                }
            }
            for r in inf.new_rows {
                if let Some(id) = cell_str(&r[0]) {
                    if !added.contains(&id) && !have.contains(&id) {
                        out.new_rows.push(r);
                    }
                }
            }
            for (k, v) in inf.via {
                out.via.insert(k, v);
            }
        }
        Ok(out)
    }

    /// Memories linked to the query's nodes and the top hits' subjects
    /// through ontology facts: inferred facts always, stated facts when
    /// the query asks about their property.
    #[allow(clippy::too_many_arguments)]
    fn inferred_expand(
        &'s self,
        sc: &Scope,
        query: &str,
        rows: &[Vec<DataValue>],
        og: &OntoGraph,
        act: &crate::ontology::Active,
        weight: f64,
        as_of: Option<i64>,
        kind: Option<&str>,
    ) -> Result<GraphOut> {
        let top_rel = rows.first().and_then(|r| r[1].get_float()).unwrap_or(0.5);
        let seeds: Vec<(String, f64)> = rows
            .iter()
            .take(INFER_FROM)
            .filter_map(|r| Some((cell_str(&r[0])?, r[1].get_float().unwrap_or(0.0))))
            .collect();
        let seed_ids: Vec<String> = seeds.iter().map(|x| x.0.clone()).collect();
        let rel_of = |id: &str| seeds.iter().find(|x| x.0 == id).map_or(0.0, |x| x.1);
        // seed nodes and their scores
        let mut seed_nodes: BTreeMap<String, f64> = BTreeMap::new();
        let bump = |m: &mut BTreeMap<String, f64>, n: String, v: f64| {
            let e = m.entry(n).or_insert(v);
            if v > *e {
                *e = v;
            }
        };
        for (id, n) in self.subjects_of(sc, &seed_ids)? {
            if og.nodes.contains(&n) {
                bump(&mut seed_nodes, n, rel_of(&id));
            }
        }
        for ((fs, _, fo), mems) in &og.evidence {
            for m in mems {
                if seed_ids.contains(m) {
                    bump(&mut seed_nodes, fs.clone(), rel_of(m));
                    bump(&mut seed_nodes, fo.clone(), rel_of(m));
                }
            }
        }
        for n in query_nodes(query, &og.nodes) {
            bump(&mut seed_nodes, n, top_rel);
        }
        if seed_nodes.is_empty() {
            return Ok(GraphOut::default());
        }
        let qprops = query_props(act, query);
        // the other end of each fact touching a seed node
        let mut reached: BTreeMap<String, (f64, String)> = BTreeMap::new();
        for (fs, fp, fo, derived) in &og.facts {
            let keep = if qprops.is_empty() { *derived } else { qprops.contains(fp) };
            if !keep {
                continue;
            }
            for (a, b) in [(fs, fo), (fo, fs)] {
                let Some(&v) = seed_nodes.get(a) else { continue };
                if a == b || seed_nodes.contains_key(b) && !qprops.contains(fp) {
                    continue;
                }
                let why = if *derived { format!("inferred: {fp}") } else { format!("linked: {fp}") };
                let e = reached.entry(b.clone()).or_insert((v, why.clone()));
                if v > e.0 || (*derived && e.1.starts_with("linked")) {
                    *e = (v.max(e.0), why);
                }
            }
        }
        if reached.is_empty() {
            return Ok(GraphOut::default());
        }
        // memories about the reached nodes: their claims, the evidence of
        // the facts about them, and memories mentioning an entity of that
        // name
        let nodes: BTreeSet<String> = reached.keys().cloned().collect();
        let mut cand: BTreeMap<String, (f64, String)> = BTreeMap::new();
        let mut add = |id: String, node: &str| {
            let (v, why) = &reached[node];
            let sc_ = weight * v;
            let e = cand.entry(id).or_insert((sc_, why.clone()));
            if sc_ > e.0 {
                *e = (sc_, why.clone());
            }
        };
        for (id, n) in self.about_nodes(sc, &nodes)? {
            add(id, &n);
        }
        for ((fs, _, fo), mems) in &og.evidence {
            for n in [fs, fo] {
                if nodes.contains(n) {
                    for m in mems {
                        add(m.clone(), n);
                    }
                }
            }
        }
        let (_, by_name) = self.entity_names(sc)?;
        let eids: Vec<String> = nodes.iter().filter_map(|n| by_name.get(n)).flatten().cloned().collect();
        if !eids.is_empty() {
            let (view, _) = self.entity_names(sc)?;
            let name_of: BTreeMap<&String, &String> =
                nodes.iter().flat_map(|n| by_name.get(n).into_iter().flatten().map(move |e| (e, n))).collect();
            let raw: Vec<String> = view.rows.keys().filter(|e| name_of.contains_key(&view.canonical(e))).cloned().collect();
            for (e, id) in self.mentioning(sc, &raw)? {
                if let Some(n) = name_of.get(&view.canonical(&e)) {
                    add(id, n);
                }
            }
        }
        let have: BTreeSet<String> = rows.iter().filter_map(|r| cell_str(&r[0])).collect();
        let mut out = GraphOut::default();
        let mut fresh: Vec<(String, f64)> = vec![];
        for (id, (v, why)) in cand {
            out.via.insert(id.clone(), why);
            if have.contains(&id) {
                out.set.insert(id, v);
            } else {
                fresh.push((id, v));
            }
        }
        fresh.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        fresh.truncate(GRAPH_MAX_NEW);
        out.new_rows = self.rows_of(sc, &fresh, as_of, kind)?;
        Ok(out)
    }

    /// Personalized PageRank over the neighbourhood of the hits.
    #[allow(clippy::too_many_arguments)]
    fn ppr_rerank(
        &'s self,
        sc: &Scope,
        query: &str,
        rows: &[Vec<DataValue>],
        og: Option<&OntoGraph>,
        gw: f64,
        as_of: Option<i64>,
        kind: Option<&str>,
    ) -> Result<GraphOut> {
        let seeds: Vec<(String, f64)> = rows
            .iter()
            .take(GRAPH_SEEDS)
            .filter_map(|r| Some((cell_str(&r[0])?, r[1].get_float().unwrap_or(0.0).max(0.0))))
            .collect();
        if seeds.is_empty() {
            return Ok(GraphOut::default());
        }
        let mut g = Graph::default();
        let m = |id: &str| format!("m:{id}");
        let seed_ids: Vec<String> = seeds.iter().map(|x| x.0.clone()).collect();
        let pool_ids: Vec<String> = rows.iter().filter_map(|r| cell_str(&r[0])).collect();
        // hop 1: entities and subject nodes of the pool
        let (view, by_name) = self.entity_names(sc)?;
        let ment = self.mentions_of(sc, &pool_ids)?;
        let mut ents: BTreeSet<String> = BTreeSet::new();
        for (id, e) in &ment {
            let c = view.canonical(e);
            g.edge(&m(id), &format!("e:{c}"), 1.0);
            ents.insert(c);
        }
        let mut onodes: BTreeSet<String> = BTreeSet::new();
        for (id, n) in self.subjects_of(sc, &pool_ids)? {
            g.edge(&m(&id), &format!("n:{n}"), 1.0);
            onodes.insert(n);
        }
        // the query's own nodes and entities
        let qnodes: BTreeSet<String> = og.map(|o| query_nodes(query, &o.nodes)).unwrap_or_default();
        let mut qents: BTreeSet<String> = BTreeSet::new();
        {
            let all_names: BTreeSet<String> = by_name.keys().cloned().collect();
            for n in query_nodes(query, &all_names) {
                if let Some(es) = by_name.get(&n) {
                    qents.extend(es.iter().cloned());
                }
            }
        }
        // ontology facts among the nodes (and their evidence)
        if let Some(og) = og {
            onodes.extend(qnodes.iter().cloned());
            for ((fs, _, fo), mems) in &og.evidence {
                for mm in mems {
                    if pool_ids.contains(mm) {
                        g.edge(&m(mm), &format!("n:{fs}"), 1.0);
                        g.edge(&m(mm), &format!("n:{fo}"), 1.0);
                        onodes.insert(fs.clone());
                        onodes.insert(fo.clone());
                    }
                }
            }
            let mut frontier: BTreeSet<String> = BTreeSet::new();
            for (fs, _, fo, derived) in &og.facts {
                if onodes.contains(fs) || onodes.contains(fo) {
                    g.edge(&format!("n:{fs}"), &format!("n:{fo}"), if *derived { 0.5 } else { 1.0 });
                    frontier.insert(fs.clone());
                    frontier.insert(fo.clone());
                }
            }
            onodes.extend(frontier);
            for ((fs, _, fo), mems) in &og.evidence {
                if onodes.contains(fs) || onodes.contains(fo) {
                    for mm in mems {
                        g.edge(&m(mm), &format!("n:{fs}"), 1.0);
                        g.edge(&m(mm), &format!("n:{fo}"), 1.0);
                    }
                }
            }
        }
        // entity ↔ ontology node bridges (an entity named like a node)
        for n in &onodes {
            if let Some(es) = by_name.get(n) {
                for e in es {
                    g.edge(&format!("e:{e}"), &format!("n:{n}"), 1.0);
                    ents.insert(e.clone());
                }
            }
        }
        ents.extend(qents.iter().cloned());
        // hop 2: the other memories of those entities and nodes (entities
        // mentioned by many memories are not expanded: they would pull in
        // everything a speaker ever said)
        let mut by_ent: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let ent_list: Vec<String> = ents.iter().cloned().collect();
        // `mentioning` takes raw ids: every row merged into a canonical one
        let raw: Vec<String> = view.rows.keys().filter(|e| ents.contains(&view.canonical(e))).cloned().collect();
        let _ = &ent_list;
        for (e, id) in self.mentioning(sc, &raw)? {
            by_ent.entry(view.canonical(&e)).or_default().push(id);
        }
        let mut mems: BTreeSet<String> = pool_ids.iter().cloned().collect();
        for (e, ids) in &by_ent {
            if ids.len() > HUB_DEGREE {
                continue;
            }
            for id in ids {
                if mems.len() >= PPR_MAX_MEMS && !mems.contains(id) {
                    break;
                }
                mems.insert(id.clone());
                g.edge(&m(id), &format!("e:{e}"), 1.0);
            }
        }
        let about = self.about_nodes(sc, &onodes)?;
        let mut node_deg: BTreeMap<&str, usize> = BTreeMap::new();
        for (_, n) in &about {
            *node_deg.entry(n.as_str()).or_default() += 1;
        }
        for (id, n) in &about {
            if node_deg[n.as_str()] > HUB_DEGREE || (mems.len() >= PPR_MAX_MEMS && !mems.contains(id)) {
                continue;
            }
            mems.insert(id.clone());
            g.edge(&m(id), &format!("n:{n}"), 1.0);
        }
        // hubs (a speaker every memory mentions, `user` in a one-user store)
        // carry no information and flatten the walk: cut them (HippoRAG
        // weighs phrase nodes by specificity for the same reason)
        let mut hubs: BTreeSet<String> = BTreeSet::new();
        for (e, ids) in &by_ent {
            if ids.len() > HUB_DEGREE {
                hubs.insert(format!("e:{e}"));
            }
        }
        for (n, d) in &node_deg {
            if *d > HUB_DEGREE {
                hubs.insert(format!("n:{n}"));
            }
        }
        g.cut(&hubs);
        // claims of one statement are one passage
        let mut by_stmt: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for id in &mems {
            by_stmt.entry(memory_recall::statement_of(id).to_string()).or_default().push(id.clone());
        }
        for ids in by_stmt.values() {
            for w in ids.windows(2) {
                g.edge(&m(&w[0]), &m(&w[1]), 1.0);
            }
        }
        // restart distribution: every pool row by relevance (HippoRAG 2
        // seeds all passages with their dense scores), plus the query's
        // nodes and entities; a row with no edges keeps its own mass
        let mut restart: BTreeMap<String, f64> = BTreeMap::new();
        let pool_rel: Vec<(String, f64)> =
            rows.iter().filter_map(|r| Some((cell_str(&r[0])?, r[1].get_float().unwrap_or(0.0).max(0.0)))).collect();
        let total: f64 = pool_rel.iter().map(|x| x.1).sum::<f64>().max(1e-9);
        for (id, v) in &pool_rel {
            g.node(&m(id));
            *restart.entry(m(id)).or_default() += v / total;
        }
        let q: Vec<String> = qnodes
            .iter()
            .map(|n| format!("n:{n}"))
            .chain(qents.iter().map(|e| format!("e:{e}")))
            .filter(|k| g.idx.contains_key(k))
            .collect();
        if !q.is_empty() {
            let each = 0.5 / q.len() as f64;
            for k in q {
                *restart.entry(k).or_default() += each;
            }
        }
        let pr = g.ppr(&restart, PPR_DAMPING, PPR_ITERS);
        // memory scores, scaled to the top relevance
        let top_rel = pool_rel.iter().map(|x| x.1).fold(0.0, f64::max).max(1e-9);
        mems.extend(pool_rel.iter().map(|x| x.0.clone()));
        let mut mscore: Vec<(String, f64)> = mems
            .iter()
            .filter_map(|id| g.idx.get(&m(id)).map(|&i| (id.clone(), pr[i])))
            .collect();
        let max_pr = mscore.iter().map(|x| x.1).fold(0.0, f64::max);
        if max_pr <= 0.0 {
            return Ok(GraphOut::default());
        }
        for x in mscore.iter_mut() {
            x.1 = x.1 / max_pr * top_rel;
        }
        let rel: BTreeMap<String, f64> =
            rows.iter().filter_map(|r| Some((cell_str(&r[0])?, r[1].get_float().unwrap_or(0.0)))).collect();
        let mut out = GraphOut { replace: true, ..Default::default() };
        let mut fresh = vec![];
        for (id, p) in mscore {
            match rel.get(&id) {
                // the walk can lift a row it supports, never sink one the
                // search found: replacing relevance with the blend let an
                // older, better-linked value outrank the current one
                // (knowledge-update questions, docs/benchmarks/memory.md)
                Some(&r) => {
                    out.set.insert(id, r.max((1.0 - gw) * r + gw * p));
                }
                None => {
                    if p > 0.0 {
                        fresh.push((id, gw * p));
                    }
                }
            }
        }
        // pool rows outside the neighbourhood keep their relevance
        for (id, r) in &rel {
            out.set.entry(id.clone()).or_insert(*r);
        }
        fresh.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        fresh.truncate(GRAPH_MAX_NEW);
        for (id, _) in &fresh {
            out.via.insert(id.clone(), "graph".to_string());
        }
        out.new_rows = self.rows_of(sc, &fresh, as_of, kind)?;
        let _ = seed_ids;
        Ok(out)
    }
}

/// A small weighted undirected graph for PPR.
#[derive(Default)]
struct Graph {
    idx: BTreeMap<String, usize>,
    adj: Vec<Vec<(usize, f64)>>,
}

impl Graph {
    fn node(&mut self, k: &str) -> usize {
        if let Some(&i) = self.idx.get(k) {
            return i;
        }
        let i = self.adj.len();
        self.idx.insert(k.to_string(), i);
        self.adj.push(vec![]);
        i
    }

    /// Remove the edges of `nodes` (they stay, with no neighbours).
    fn cut(&mut self, nodes: &BTreeSet<String>) {
        let ids: BTreeSet<usize> = nodes.iter().filter_map(|k| self.idx.get(k).copied()).collect();
        if ids.is_empty() {
            return;
        }
        for (i, a) in self.adj.iter_mut().enumerate() {
            if ids.contains(&i) {
                a.clear();
            } else {
                a.retain(|(j, _)| !ids.contains(j));
            }
        }
    }

    fn edge(&mut self, a: &str, b: &str, w: f64) {
        if a == b {
            return;
        }
        let (i, j) = (self.node(a), self.node(b));
        self.adj[i].push((j, w));
        self.adj[j].push((i, w));
    }

    /// Power iteration: r = (1 − d)·p + d·Pᵀr, P row-normalised; mass at
    /// dangling nodes returns to the restart distribution.
    fn ppr(&self, restart: &BTreeMap<String, f64>, d: f64, iters: usize) -> Vec<f64> {
        let n = self.adj.len();
        let mut p = vec![0.0; n];
        let mut tot = 0.0;
        for (k, v) in restart {
            if let Some(&i) = self.idx.get(k) {
                p[i] += v;
                tot += v;
            }
        }
        if tot <= 0.0 {
            return vec![0.0; n];
        }
        for x in p.iter_mut() {
            *x /= tot;
        }
        let deg: Vec<f64> = self.adj.iter().map(|a| a.iter().map(|x| x.1).sum()).collect();
        let mut r = p.clone();
        for _ in 0..iters {
            let mut next = vec![0.0; n];
            let mut dangling = 0.0;
            for i in 0..n {
                if r[i] == 0.0 {
                    continue;
                }
                if deg[i] <= 0.0 {
                    dangling += r[i];
                    continue;
                }
                for &(j, w) in &self.adj[i] {
                    next[j] += d * r[i] * w / deg[i];
                }
            }
            let mut delta = 0.0;
            for i in 0..n {
                let v = next[i] + (1.0 - d + d * dangling) * p[i];
                delta += (v - r[i]).abs();
                r[i] = v;
            }
            if delta < 1e-8 {
                break;
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{Active, OntologyDoc, PERSONAL};

    #[test]
    fn query_matching() {
        let d: OntologyDoc = serde_json::from_str(PERSONAL).unwrap();
        let act = Active::merge(&[d], "cardinality", "decide");
        let p = query_props(&act, "Who are my relatives?");
        assert!(p.contains("relative_of"), "{p:?}");
        let p = query_props(&act, "where does Anna live now?");
        assert!(p.contains("lives_in"), "{p:?}");
        assert!(!query_props(&act, "what did we eat").contains("same_as"));
        let nodes: BTreeSet<String> = ["user", "user.sister", "anna"].iter().map(|x| x.to_string()).collect();
        let q = query_nodes("Where does my sister live? Ask Anna.", &nodes);
        assert_eq!(q, nodes);
    }

    #[test]
    fn ppr_prefers_neighbours() {
        let mut g = Graph::default();
        g.edge("m:a", "e:x", 1.0);
        g.edge("m:b", "e:x", 1.0);
        g.edge("m:c", "e:y", 1.0);
        g.edge("m:d", "e:y", 1.0);
        let r = g.ppr(&[("m:a".to_string(), 1.0)].into_iter().collect(), 0.5, 50);
        let at = |k: &str| r[g.idx[k]];
        assert!(at("m:b") > at("m:c"));
        assert_eq!(at("m:c"), 0.0);
        let s: f64 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "{s}");
    }
}
