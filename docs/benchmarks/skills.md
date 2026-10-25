# Skill library benchmarks

*Measured 2026-10-25 on an M3 (16 GB) shared with another agent's builds, store
and models on the USB SSD. Code: `kleos-core/examples/skill_bench.rs` and
`scripts/skills/bench_prep.py`, skill-library code as of commit `52cc81e`
(branch `wave7/skills`). Data: `.bench-data/skills/` (not in git). The runs
were cut short on request: the routing numbers are one complete run with a
6,000-token completion budget for the LLM (see the caveat); the learning and
optimizer demo is a partial sample.*

## Routing at scale (replication of "Lost in Implicit Routing")

**Setup.**
- Library: 1,855 unique public skills (SKILL.md, deduplicated by name and
  description) from 11 GitHub repositories (anthropics/skills,
  obra/superpowers, K-Dense-AI/claude-scientific-skills,
  alirezarezvani/claude-skills, ComposioHQ/awesome-claude-skills,
  wshobson/agents, huggingface/skills, openai/skills, trailofbits/skills,
  hashicorp/agent-skills, vercel-labs/agent-skills).
- 30 **target** skills with distinctive descriptions (word overlap < 0.35
  with every other skill, at most 5 per repository), 2 user tasks each (60
  tasks) written by the LLM without the skill's name or distinctive phrases.
- 1,000 **distractors** sampled from the rest, excluding anything whose
  description overlaps a target's (word Jaccard ≥ 0.3); the first 200 form the
  mid-size library. Library sizes 30 / 230 / 1,030 are the same index read
  through scope filters (targets in `default`, distractors in `d1`, `d2`).
- kleos: built-in EmbeddingGemma 2 (Metal) for the index, `find` with
  default settings and prior router weights (no outcomes, so no learning).
- LLM: Synthetic `syn:small:text` (GLM-4.7-Flash, a reasoning model),
  temperature 0, the same prompt for both systems ("which ONE skill should
  be loaded … exact name, or NONE").
- Six public skills were quarantined by the scanner at import (pipe-to-shell,
  environment exfiltration, credential paths, instruction overrides) and are
  not served, so the largest library holds 1,000 live skills.

| Library | kleos Hit@1 | kleos Recall@5 | kleos Recall@10 | MRR | LLM over kleos top-5 | LLM over the flat list | flat list size |
|---|---|---|---|---|---|---|---|
| 30 | 0.967 | 1.000 | 1.000 | 0.983 | **0.983** | 0.983 | 2.3k tokens |
| 230 | 0.850 | 0.983 | 1.000 | 0.911 | — | 0.867 (11.7% no pick) | 16.6k tokens |
| 1,000 | 0.717 | 0.917 | 0.950 | 0.798 | **0.850** (8.3% no pick) | **0.067** (80% no pick) | 71k tokens |

- **The flat list collapses** from 98% to 87% to 7% correct selection as the
  library grows, as the literature reports (73.6% → 27.4% loaded-skill recall
  at 1,000 distractors). **Pre-selection holds**: the LLM choosing from kleos's
  top 5 stays at 85% at 1,000 skills, about what kleos's Recall@5 (92%)
  allows.
- **Caveat on the 1,000-skill flat number.** The LLM had a 6,000-token
  completion budget. With a 60k-token list, the reasoning model often spent
  the whole budget reasoning and returned no answer (a probe of one task:
  `finish_reason: length`, empty content); this run counted "NONE" and "no
  answer" together (80%). A rerun with a 16,000-token budget and the two
  counted apart was started and stopped before it finished. The flat list is
  therefore worse here partly through cost and latency (71k prompt tokens per
  task) rather than wrong picks alone; kleos's answer costs about 400 tokens.
- `find` latency at 1,000 skills (no concurrent LLM load): p50 131 ms, p95
  177 ms, including the query embedding. Index build: 2,420 s for 1,030
  skills (~2.3 s per skill, dominated by EmbeddingGemma on the shared GPU;
  at most 24 chunks per skill); peak RSS 0.6 GB.
- Misses at 1,000 are mostly near neighbours (`python-packaging` →
  `python-project-structure`, `grafana-dashboards` →
  `prometheus-configuration`) and one long, generic distractor
  (`diagramming-code`) that wins several unrelated tasks through its many body
  chunks. The router is untrained in this run; outcome feedback is what it
  learns from.

## Learning loop and optimizer (partial sample)

Toy suite (`skill_bench learn`): two task families with rules no model
knows (a fictional Acme shipping rate card and an Acme SKU format), 14
successful traces and one failed one, `syn:small:text` as drafter, agent and
optimizer.

| Run | Result |
|---|---|
| Learn, run 1 | 2 clusters found (15 traces); both drafted skills were correct in content (rate card, rounding, express, discount; the SKU rules) and admitted: held-out score with the skill 0.25 and 0.33 vs 0.00 without (margin 0.2). Its answer checker then accepted one false positive (a substring match), since fixed (`answer_matches`, token boundaries and numbers). |
| Learn, run 2 (fixed checker) | Shipping skill drafted and admitted (0.25 vs 0.00 on 4 held-out tasks); the SKU draft came back without frontmatter and was rejected. |
| Optimizer, run 1 | A hand-written shipping skill missing three rules: 4 rounds, **no usable edits** (the optimizer's answers did not parse into bounded edits), so no version was created and the original stayed live (base and best 0.00). The prompt now names the valid line range and the raw answer is recorded for diagnosis; the rerun was stopped. |

What the sample shows: harvesting, clustering, drafting, the scan, the
with/without gate, activation and the provenance chain work end to end on a
real model, and the gate admits only skills that beat the baseline; the
learned skill's low absolute score comes from an ambiguity in the drafted
steps (where the 10% discount applies), which is the optimizer's job. A
verified optimizer improvement on the real model was **not** obtained in the
time allowed. The optimizer loop itself (reflection, bounded edits,
rejected-edit buffer, strict held-out gate, lineage, activation of the
winner) is verified end to end against a deterministic mock endpoint in
`kleos-core/tests/skills.rs` (`learning_loop_and_optimizer_with_a_mock_llm`:
0.0 → 1.0 held-out, the winning version supersedes with `origin: optimized`
and its edits recorded).

## Reproduce

```bash
# data (downloads public repos to .bench-data/skills/src first; see SOURCES.md there)
KLEOS_ENV_FILE=.env python3 -I kleos/scripts/skills/bench_prep.py .bench-data/skills
# kleos only (no LLM): omit KLEOS_ENV_FILE
cargo run --release -p kleos --features infer-metal --example skill_bench -- routing .bench-data/skills out.json
# with the LLM baselines (BENCH_SIZES=30,1030 to limit the sizes)
KLEOS_ENV_FILE=.env cargo run --release -p kleos --features infer-metal --example skill_bench -- routing .bench-data/skills out.json
KLEOS_ENV_FILE=.env cargo run --release -p kleos --features infer-metal --example skill_bench -- learn .bench-data/skills learn.json
```

The API key is read from `KLEOS_ENV_FILE` at run time and kept in the
process environment only; it is never printed or written.
