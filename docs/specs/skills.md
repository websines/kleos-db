# Skill library (procedural memory)

*Status: Phase 0 spec, 2026-10-25, branch `wave7/skills`. Design basis:
`research/skills-research.md` (parent repo), "Recommendations for kleos".*

A **skill** is an Agent Skills folder (`SKILL.md` with YAML frontmatter
`name` + `description`, plus `scripts/`, `references/`, `assets/`). kleos
stores skills as immutable, content-addressed versions; serves the right
ones for a task (pre-selection, not a flat list); measures every version
against real outcomes; learns new skills from successful runs and improves
existing ones with a gated optimizer; and rolls back, retires and
quarantines on evidence. Everything is reachable through one JSON entry
point, `skill_call(op, args)` (`::skill <op> {…}`, Rust, Python, TS, HTTP,
MCP, CLI).

## Model

| Concept | Rule |
|---|---|
| Version | Immutable: verbatim `SKILL.md` + files, per-file `sha256:<hex>` digests and a manifest digest (SHA-256 of the canonical manifest). Never edited; a change is a new version with `parent`. |
| Head | Which version is live, per `(scope, name)`, over **valid time** (`kleos_skill_head`, a `Validity`-keyed relation). Activating asserts a new head; the old one ends. Time travel: "which version was live when run X ran". |
| Status | Per version: `candidate`, `active`, `superseded`, `retired`, `quarantined`, `rejected`. Only `active` is served by default. |
| Scope | String, the tenancy namespace of the skill (`default`, `agent:<a>`, `team:<t>`, `ws`, `org`, `au:<a>:<u>`, …). Precedence narrowest first (run > agent_user > user > agent > team > workspace/ns > org); a narrower same-name skill shadows a broader one. |
| Actor | JSON `{human, agent, run}` on every version, head change, outcome and log row; `null` fields today (one implicit actor). Wiring to `wave7/tenancy` = filling it from the `PrincipalContext`. |
| Provenance | `origin` (`authored`, `imported`, `induced`, `optimized`, `promoted`, `installed`) + the traces, runs, memories and source path it came from. |
| Stats | Per version, from recorded outcomes: uses, successes, failures, latency, tokens, cost; Beta posterior of the success rate. |

Schema: a separate relation set (`kleos_skill_*`), created on first use, with
its own version in `kleos_skill_meta` (skill schema v1). It does not touch the
memory schema (v8; tenancy takes v9), so the two branches merge without a
version clash. Under tenancy partitions the relations route per workspace
like every other relation.

## Retrieval

`find(task, budget?, k?, scope?, scopes?, status?, disclosure?)`: FTS (BM25)
over description and body chunks + filtered HNSW over their embeddings (when
an embedder is configured), scope-filtered inside both indexes; per-skill
features (lexical, description cosine, best body-chunk cosine, name overlap,
success posterior, usage, scope narrowness) scored by a **learned linear
router** (logistic, online SGD from outcomes linked to the `route` id the
search returned). Progressive disclosure under a token budget:
descriptions for the ranked list, the body of the top hit(s) when asked and
when it fits, resources as URIs only.

## Lifecycle and learning

- `record_outcome(ver | skill, run?, success, metrics)`; flows that load a
  skill (`skill` step action `use`, or `meta.skills` on start) record their
  outcome automatically when they finish.
- **Auto-rollback**: when a version's posterior success rate is below its
  predecessor's with P > 0.95 after ≥ `min_samples` outcomes.
- **Auto-retire**: unused for `retire_unused_days`, or consistently failing.
- **Quarantine**: the static scanner flags a version; it is never served.
- **Undo**: one step, re-asserts the previous head.
- `skill.learn` (durable workflow, cron-capable, off unless an LLM endpoint
  is configured): harvest successful traces → cluster → draft via the LLM
  hook → scan → verify with vs without on held-out tasks (margin) →
  activate (`auto`) or propose (`review`).
- `skill.optimize`: bounded edit operators proposed from failure traces
  (GEPA-style reflection), ACE-style itemized deltas, strict held-out gate,
  rejected-edit buffer, edit budget; a winning version supersedes.

## Security

Static scanner (prompt injection, hidden Unicode, exfiltration/network/shell
danger, secrets, undeclared permissions); permissions declared under
`metadata`; digest pinning; detached Ed25519 signatures in minisign format
(offline, `ed25519-dalek`); scripts are stored and served, never executed by
kleos; `allowed-tools` is never honoured for MCP-served skills.

## Sharing

Promotion agent → team → workspace → org is an audited copy (new version,
`origin: promoted`) that needs approval unless the policy allows it; signed
packages (`.kleos-skill.json`) move skills between stores; install verifies
signature, digests and scan.

## Surfaces

`::skill <op>`, `Db::skill_call`, HTTP `POST /skill/:op` + registry
endpoints, MCP SEP-2640 (`skills/list` top-k partial, `skills/get`,
`skill://` resources, `resources/directory/read`) plus tools, `kleos skill …`,
Python `skill_call`, TS `kleos.skill()`.
