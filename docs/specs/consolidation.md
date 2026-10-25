# Consolidation: entity profiles and session summaries

*UPGRADE_PLAN E7. Code: `kleos-core/src/memory_consolidate.rs`. Tests:
`kleos-core/tests/memory_consolidate.rs`. Results: "Results" below.*

Consolidation folds a namespace's episodes into derived knowledge. It needs no
LLM: profiles are read off the ontology and the typed claims, summaries are
extractive. Raw memories stay the source of truth.

## What it builds

**Entity profiles** (`kleos_mem_profile {ns, node => name, text, facts, evidence, sig, updated}`):
for every ontology node (a person, `user`, a relative key such as
`user.sister`, a named thing) with at least `min_facts` (default 2) current
facts:

- the stated ontology facts about it, both directions (`sister_of: User`,
  `← sister_of: Anna`), node labels where known;
- the typed claims whose subject it is, grouped by property (`lives_in: I live
  in Lisbon now (since 2026-01-06)`);
- each value with its valid-from and the memories behind it (`evidence`);
- single-valued properties keep only the newest current value, so a profile
  follows supersession on the next run;
- typed claims count when their type is cue- or rule-backed, or a prototype
  match at a confident score (≥ 0.55, wave7/closeout; weaker prototype matches
  stay out).

**Entity profiles from the extractor** (wave7/closeout): a named entity
(GLiNER, or `entities` passed to `remember`) that 3 to 150 current memories
mention, and that no ontology profile covers (fewer than `min_facts` facts),
gets a profile of what those memories say: by property when a claim is typed,
`mentioned` otherwise, newest first, the entity's kind in the title
(`Profile: Max (pet) — mentioned: …`). An entity mentioned by more than 150
memories (a speaker of every turn) is a hub and gets none. Chat stores, where
the ontology knows little beyond the speakers (24 profiles over LoCoMo's 10
conversations before), get profiles for the people, pets and places they talk
about. The count on LoCoMo and QA with profiles on vs off were **not
re-measured** (measurement cut short by the user).

The text is one line per node:

```
Profile: User — employer: I work at Siemens as an engineer (since 2026-01-04); lives_in: I live in Lisbon now (since 2026-01-06); ← sister_of: Anna (since 2026-01-03)
```

**Session summaries** (memories of kind `summary`, role `other`, source
`summary:<group>`, plus `kleos_mem_summary {ns, sid => grp, sources, method, sig, at}`):
for every source group (the source up to its first `#`: a conversation, a
session) with at least `min_claims` (default 8) declarative claims:

- the claims are ranked by centrality (cosine to the group's centroid; word
  overlap without an embedder) and picked with MMR (λ 0.7) against redundancy,
  `min(max_summary, max(2, ⌈n/5⌉))` of them (default `max_summary` 6);
- the summary is those claims in time order: `Summary of <group> (<first> – <last>): …`;
- its valid-from is the group's newest claim, it has its own vector, and it is
  found by `recall` like any memory (`kind: '!summary'` leaves it out);
- **replaced values are left out** (wave7/closeout): a claim a newer one may
  supersede (a pending `supersedes` decision), and an older value of a
  single-valued property when a newer current claim of the same subject and
  property is similar to it (cosine ≥ 0.6; stores with vectors only). Superseded
  claims were already out (summaries read current memories). The run reports
  `replaced_dropped`.

**Optional abstractive summaries:** `llm: {url, model, api_key_env,
max_tokens}` (an OpenAI-compatible endpoint; the key is read from the named
environment variable, never stored) rewrites each summary from its extracted
claims. Off by default; needs the `requests` feature; a failed call keeps the
extractive text (`llm_errors`).

## Guarantees

- **Never authoritative.** Summaries have role `other`, so they never take
  part in supersession or duplicate decisions, and nothing supersedes a raw
  memory because of them. Profiles are separate rows, not memories.
- **Rebuilt on change.** Each item carries a signature of its inputs (the
  source ids, the method; the profile text). A run rebuilds what changed,
  leaves the rest (`unchanged`), and retracts summaries of groups that no
  longer qualify and profiles of nodes that no longer do.
- **Erased with their sources.** `forget {mode: 'erase'}` erases every summary
  built from an erased memory (summary memory, vector, provenance) and drops
  every profile citing it; the next run rebuilds them without it. A plain
  retraction leaves them until the next run.

## Running it

- `::memory consolidate {ns, profiles?, summaries?, min_claims?, max_summary?, min_facts?, llm?}`
  (Rust `Db::memory_consolidate`, Python `memory_consolidate(**opts)`, HTTP
  `POST /memory/consolidate`) runs it now and returns counts, token totals and
  time.
- **Automatically**, as the durable workflow `memory.consolidate` (built-in
  step `consolidate`, retried, resumable like any `::flow` workflow):
  - `every: N` starts it after every N new memories of the namespace (counted
    in-process after each `remember`; one run per namespace at a time, lane
    `consolidate:<ns>`);
  - `cron: '@every 1h'` (any `::flow` schedule) runs it on a schedule.

  **On by default:** a namespace with no setting consolidates after every
  `DEFAULT_CONSOLIDATE_EVERY` (200) new memories; `every: 0` turns it off.
  Runs go to the flow queue `consolidate`, and the store runs them itself: a
  background thread of the process (`DbInstance::consolidate_kick`), started
  lazily after a `remember` that queued a run, ticks that queue until nothing
  is due and exits. A flow worker (`flow_worker`, `kleos server
  --flow-worker`) or `::flow tick` can run them too; leases keep one run from
  executing twice. Profiles and summaries are shown in the context block by
  default once built (the user's choice; about +38% context tokens on LoCoMo,
  QA +1.9 points there and −0.8 on LongMemEval, docs/benchmarks/memory.md).

## Context output

`recall {format: 'context'}` puts the profiles of the hits' subjects and
mentioned entities ahead of the memories, within a quarter of the budget:

```
Profiles (current facts per person or thing):
- Profile: User — lives_in: I live in Lisbon now (since 2026-01-06); …

Recalled memories, grouped by topic, …
```

`profiles: false` turns it off. `RecallContext.profiles` lists the lines.

## Results

See "Graph recall and consolidation" in `docs/benchmarks/memory.md`. In short:
summaries in the context block +6 QA net on LoCoMo (multi-hop 0.673 → 0.727),
neutral on LongMemEval (knowledge-update −2: a summary can quote a replaced
value); profiles within noise on both; 7.7× / 4.6× compression; about 3 s per
LoCoMo conversation, no LLM.
