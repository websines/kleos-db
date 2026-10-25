# Skill library (procedural memory)

*Status: built, 2026-10-25, branch `wave7/skills`. Design basis:
`research/skills-research.md` (parent repo), "Recommendations for kleos".
Code: `kleos-core/src/skill.rs` (model, storage, interop, lifecycle,
outcomes), `skill_find.rs` (retrieval, router, SEP-2640 entries),
`skill_scan.rs` (scanner), `skill_sign.rs` (signatures), `skill_share.rs`
(packages, registry, promotion), `skill_learn.rs` (learning loop, optimizer);
`kleos-bin/src/mcp.rs` (skills over MCP), `cli.rs` (`kleos skill`),
`server.rs` (`/skill/:op`, `/skills/registry`). Tests:
`kleos-core/tests/skills.rs`, `kleos-bin/tests/mcp_skills.rs`. Benchmark:
`kleos-core/examples/skill_bench.rs`, docs/benchmarks/skills.md.*

A **skill** is an Agent Skills folder (agentskills.io): `SKILL.md` with YAML
frontmatter (`name`, `description`, optional `license`, `compatibility`,
`metadata`, `allowed-tools`) and a Markdown body, plus `scripts/`,
`references/`, `assets/`. kleos adds what a folder cannot have:

1. **Outcome-measured, self-correcting versions.** Every version is immutable
   and content-addressed; outcomes of real runs give it a success rate; a
   version that does worse than its predecessor is rolled back, a skill that
   keeps failing or is never used is retired.
2. **Pre-selection instead of a flat list.** `find` returns the few skills
   that fit a task (body-aware hybrid search, a learned router, a token
   budget). Routing over a flat list of descriptions collapses as a library
   grows (1,000 distractors: 73.6% → 27.4% loaded-skill recall in the
   literature; our replication is in docs/benchmarks/skills.md).
3. **Learning behind a gate.** New skills are drafted from successful runs
   and optimized with bounded edits, and admitted only when replaying
   held-out tasks shows they beat the baseline.
4. **Governed sharing.** Scopes, audited promotion with approval, signed
   packages, a static scanner, digest pinning; scripts are never executed.

Everything goes through one JSON entry point, `skill_call(op, args)`:
`::skill <op> {…}` in CozoScript, `Db::skill_call` / `DbInstance::skill_call`
in Rust, `CozoDbPy.skill_call` in Python, `kleos.skill(op, args)` /
`findSkills` / `recordSkillOutcome` in TypeScript, `POST /skill/:op` over
HTTP, `kleos skill <op>` on the command line, and MCP (SEP-2640 plus tools).

## Model

| Concept | Rule |
|---|---|
| **Version** | Immutable. The verbatim bytes of every file, a manifest of `{path, digest: "sha256:<hex>", size}` per file, and the **version digest** = SHA-256 of the canonical manifest JSON `{"files":[{"digest","path","size"},…]}` (sorted by path). Identical content in the same scope is the same version. A change is a new version with `parent`. |
| **Head** | Which version of `(scope, name)` is live, over **valid time** (`kleos_skill_head`, keyed by a `Validity`). Activating, rolling back, retiring, quarantining and undoing each assert a new head; the previous one ends. `as_of {name, at}` answers "which version was live at T". |
| **Status** | Per version: `candidate` (stored, not served), `active` (live), `superseded`, `retired`, `quarantined` (the scanner flagged it; never served), `rejected` (failed verification). Only active skills are indexed and served. |
| **Scope** | A string, the tenancy namespace of the skill: `default` today; `run:<a>:<u>:<r>`, `au:<a>:<u>`, `user:<u>`, `agent:<a>`, `team:<t>`, `ws`, `org` under tenancy (any other string is a legacy namespace, workspace rank). Precedence narrowest first: run > agent_user > user > agent > team > workspace > org. A narrower skill shadows a broader one of the same name. Calls take `scope` or `scopes` (the visible set). |
| **Actor** | `{human, agent, run, chain?}` on every version, head change, outcome, trace and log row. Null fields today (one implicit actor); under tenancy it is filled from the `PrincipalContext`. |
| **Provenance** | `origin`: `authored`, `imported`, `induced` (learned), `optimized`, `proposed`, `promoted`, `installed`; plus the source folder, traces, held-out cases, promotion, approver, package digest and signers, and declared permissions. |
| **Stats** | Per version, from outcomes: uses, successes, failures, success rate, Beta(1+s, 1+f) posterior mean, mean latency and tokens, total cost, last use. |

### Relations

A separate relation set, created on first use (skill schema v1, recorded in
`kleos_skill_meta.version`). It never touches the memory relations (memory
schema v8; tenancy takes v9), so the two branches merge without a version
clash. Under tenancy partitions the relations route per workspace like any
other relation.

| Relation | Key ⇒ value |
|---|---|
| `kleos_skill_meta` | `{key ⇒ value: Json}`: version, embedder, lifecycle, policy, learn settings, router weights |
| `kleos_skill` | `{scope, name ⇒ ver?, status, description, created, updated, actor}`: the catalog |
| `kleos_skill_ver` | `{ver ⇒ scope, name, seq, digest, frontmatter, body, files, parent?, origin, edits?, actor, provenance, scan, created}` |
| `kleos_skill_vstat` | `{ver ⇒ status, at, reason?}` |
| `kleos_skill_head` | `{scope, name, vt: Validity ⇒ ver?, status, actor, reason?}` |
| `kleos_skill_blob` | `{digest ⇒ bytes}`: content-addressed file store (shared across versions and scopes) |
| `kleos_skill_chunk` (+ FTS) / `kleos_skill_vec` (+ HNSW) | `{scope, name, n ⇒ …}`: index of live versions only |
| `kleos_skill_outcome` | `{ver, oid ⇒ scope, name, at, run?, success, latency_ms?, tokens?, cost?, model?, task?, source, route?, actor, metrics?}` |
| `kleos_skill_use` | `{run, ver ⇒ scope, name, at, route?}`: skills a workflow used |
| `kleos_skill_trace` | `{tid ⇒ scope, task, transcript, success, output?, run?, at, actor, metrics?, skills}` |
| `kleos_skill_eval` | `{scope, name, case ⇒ task, expected?, check?, split, source, at}` |
| `kleos_skill_log` | `{scope, name, at, eid ⇒ op, ver?, from?, actor, detail?}`: full history |
| `kleos_skill_promo` | `{id ⇒ ver, scope, name, to, status, actor, approver?, at, decided?, reason?, new_ver?}` |
| `kleos_skill_key`, `kleos_skill_sig` | trusted keys; detached signatures per version |
| `kleos_skill_pub` | `{name, ver ⇒ scope, digest, package, at}`: this store's registry |
| `kleos_skill_route`, `kleos_skill_run` | search logs (router training); learn/optimize run state |

## Interop

- **Import** `import {path, scope?, activate?}`: a skill folder, a folder of
  skill folders, or a Claude Code plugin (`.claude-plugin/plugin.json`, its
  `skills` field or `skills/`). agentskills.io validation: name rules (1-64
  lowercase letters, digits, single hyphens; must match the folder),
  non-empty description ≤ 1024 characters, compatibility ≤ 500; SEP-2640
  limits (512 files, 16 MiB). Hidden files and symlinks are skipped. Imported
  skills start as candidates unless `activate` (and the scan is clean).
- **Export** `export {path, name? | all active, format: folder | plugin,
  annotate?}`: the exact stored bytes, so import → export is lossless (the
  test round-trips the repository's skills byte for byte). `annotate` adds
  `kleos.version`, `kleos.digest`, `kleos.scope`, `kleos.success_rate`,
  `kleos.uses` under the frontmatter `metadata` (namespaced, so other
  clients ignore them). `plugin` also writes `.claude-plugin/plugin.json`.
- **Author** `put {skill_md, files?: {path: text | {b64}}, scope?, parent?,
  activate?}`; agents use `propose` (never activates).

## Retrieval

`find {task, k? (5), budget? (2000 tokens), scope? | scopes?, disclosure?
(descriptions | body | auto), explore?, log? (true)}`:

1. **Candidates**: BM25 over the description and body chunks of every live
   skill (`kleos_skill_chunk:fts`) and filtered HNSW over their embeddings
   (`kleos_skill_vec:hnsw`, when an embedder is attached), the scope filter
   running inside both indexes; up to 200 per leg, 120 skills kept.
2. **Exact features** per candidate skill: normalized lexical score, cosine to
   the description chunk, best cosine to any body or reference chunk (the body
   carries most of the routing signal), name-word overlap with the task,
   posterior success rate (centred), usage (log), scope narrowness.
3. **Learned router**: `sigmoid(w · f)`. Prior weights favour the semantic
   legs; every outcome reported with the search's `route` id is one online
   SGD step (logistic loss, L2): the used skill is a positive on success, a
   negative on failure, and on success the skills ranked above it are weak
   negatives (weight 0.3). `explore` adds a Beta-posterior UCB bonus for
   skills with few outcomes. Weights live in `kleos_skill_meta.router`
   (`config {router: 'reset'}`).
4. **Shadowing and disclosure**: one skill per name (narrowest scope); then
   descriptions for the ranked list while they fit the budget, the top hit's
   body when `disclosure: 'body'` (or `auto` and the top score clearly
   leads) and it fits, resources as URIs only.

The answer: `{route, skills: [{name, scope, ver, uri, description, score,
rank, stats, features, tokens, body?, resources}], tokens, budget}`.

Embeddings: `init {embedder}` (default the memory store's embedder, e.g.
EmbeddingGemma 2; queries use its `_query` twin). Without an embedder `find`
is lexical only. Index chunks: the description, then the body cut at
headings and paragraphs (≤ 1,000 characters, at most 24 chunks per skill),
then the first chunk of up to four text references.

## Outcomes

- `record_outcome {ver | name+scope, success, run?, route?, latency_ms?,
  tokens?, cost?, model?, task?, metrics?, source?}`.
- **Automatic capture from `::flow`**: a workflow step `{"skill": {"op":
  "use", "name": …}}` (or `skill use {run}` from a code-mode worker, or
  `meta.skills` on `flow start`) links the workflow to the version; when it
  finishes, kleos records `success = completed`, the latency since start, and
  the route if one was given (`source: flow`).
- `stats {ver | name}` (with the predecessor's), `record_trace` / `traces`
  (transcripts for the learning loop), `eval_add` / `evals` / `eval`.

## Lifecycle

| Mechanism | Rule |
|---|---|
| Auto-rollback | After each outcome of a live version: when both it and its predecessor have ≥ `min_samples` (20) outcomes and P(p_new < p_old) > `rollback_confidence` (0.95) under Beta(1+s, 1+f) posteriors (normal approximation), the predecessor is re-activated and the new version retired, with the evidence in the reason. |
| Auto-retire | `lifecycle` (also the last step of every `skill.learn` run): retire live skills unused for `retire_unused_days` (90), or with ≥ `min_samples` outcomes and P(p < `retire_below` (0.25)) > 0.95. |
| Quarantine | A version the scanner flags is stored as `quarantined` and can never be activated; `quarantine {ver}` takes a live version out of service. `release {ver, confirm: true}` is the explicit override. |
| Undo | `undo {name}` re-asserts the previous head (it reverts an activation, rollback, retirement or quarantine). `rollback {name, to?}` goes to the previous live version (or `to`). |
| History | `history {name}`: versions with stats, the head timeline and the log. `erase {name, confirm: true}` deletes everything of a skill. |

Settings: `config {lifecycle: {auto_rollback, min_samples,
rollback_confidence, retire_unused_days, retire_below}}`.

## Learning loop (`skill.learn`)

A durable built-in workflow on the flow queue `skills` (steps `harvest`,
`propose`, `verify`, `admit`, `maintain`, each a `skill` step action, so a
crash resumes at the next step). Started by `learn {scope? | scopes?,
wait?}` (with `wait` it runs inside the call) or by its schedule.
**Zero-maintenance default**: `config {learn: {llm: {url, model,
api_key_env}}}` schedules it `@daily` (`cron`, `off` for manual runs only);
with no LLM endpoint there is no schedule and `learn` is refused.

1. **Harvest** successful and failed traces of the last `since_days` (30):
   `record_trace` rows, completed workflows that carry `meta.task`, and
   `trace` memories of the namespace that is the scope.
2. **Cluster** tasks by embedding cosine (else word overlap) at
   `cluster_threshold` (0.6). A cluster needs `min_support` (3) successes and
   is skipped when a live skill already covers it (a confident `find` hit
   with posterior ≥ 0.6) or its traces already fed a skill. At most
   `max_new` (3) clusters per run; `holdout` (0.4) of each cluster's
   successes are held out.
3. **Propose**: the LLM drafts one focused SKILL.md per cluster from its
   training traces (Steps with every rule and constant, Pitfalls from the
   failed runs, Check; under 450 words; SkillsBench: 2-3 focused modules beat
   documentation-style skills). The best trace is attached as
   `references/example.md`. Stored as a candidate (`origin: induced`) with
   its traces as provenance; the scanner runs. The held-out and training
   traces become the candidate's eval set.
4. **Verify**: replay the held-out tasks with the LLM as the agent, with the
   skill in its instructions and without any skill. A case passes on an exact
   check against the recorded result (`ANSWER:` line, normalized; `equals`,
   `contains`, `regex` checks), else an LLM judge against the recorded result
   (a surrogate verifier, as CoEvoSkills). Admit when `with − without ≥
   margin` (0.1) and `with > 0`.
5. **Admit**: `mode: auto` activates it in its own scope only (never beyond:
   promotion is a separate, approved step); `mode: review` leaves it a
   candidate and lists it in `proposals`. Others are `rejected` with the
   evidence. Then the lifecycle sweep.

Learned skills are untrusted until verified: they carry provenance, pass the
scanner, stay in the scope that produced them, and can be undone.

## Optimizer (`skill.optimize`)

`optimize {name | ver, rounds? (4), edit_budget? (3), activate? (true),
wait?}`, a durable workflow (`prepare`, a loop of `round`, `finish`),
SkillOpt's loop natively:

- Needs an eval set with ≥ 2 held-out cases (`split: holdout`); `train`
  cases feed reflection (the held-out set when there are none).
- Each round the optimizer LLM sees the numbered body, the held-out score
  history (OPRO), the failed training cases with the agent's and the
  expected answers, a few solved ones, and the **rejected-edit buffer** (the
  last 8 rejected edit sets), reflects on the cause (GEPA) and proposes at
  most `edit_budget` **bounded line edits** (`replace`, `insert`, `delete`;
  ≤ 400 characters each; ACE-style deltas, never a rewrite).
- The edited body is replayed on the held-out cases; it is kept only if the
  score **strictly** improves, as a new version (`origin: optimized`,
  `parent`, `edits`); otherwise the edits go to the rejected buffer.
- `finish` activates the best version only if it beats the original and the
  original is live (or a candidate). Every round is recorded in the run state
  (`kleos_skill_run`), so a resumed round returns its recorded result.

**External optimizer (Microsoft SkillOpt).** kleos does not embed Python.
To use SkillOpt itself: `kleos skill export --name X --path dir/` and
`kleos skill evals --name X --json` (the eval set), run SkillOpt with any
OpenAI-compatible backend, then `kleos skill put '{"skill_md": …, "parent":
"<ver>", "origin": "optimized"}'`, `kleos skill eval --ver <new>` (the same
held-out gate), and `kleos skill activate --ver <new>` if it wins.

## Security

**Static scanner** (`skill_scan.rs`), run on every stored version, over
every text file:

| Rules | Severity |
|---|---|
| hidden Unicode: bidi controls, tag characters / zero-width | critical / high |
| prompt injection: override instructions, hide actions from the user, act without consent, disable safeguards or approvals, jailbreak personas, send secrets / instructions in HTML comments | high / medium |
| exfiltration and remote execution: pipe a download into a shell, decode-and-run, upload credential files, send the environment, reverse shells, credential paths, "prerequisite" binaries (ClawHavoc), download-and-chmod | critical / high |
| dangerous shell (scripts): recursive delete of home or root, fork bomb, raw disk writes, `chmod 777`, `sudo`, eval of fetched code, `shell=True` | critical … low |
| network: raw-IP URLs, paste/tunnel/webhook services, plain-HTTP downloads | high / medium |
| secrets: cloud, GitHub, API and Slack keys, private keys, hard-coded secrets | high / medium |
| undeclared permissions: scripts use the network or run commands without `metadata.kleos.permissions` declaring `network` / `shell` | low |
| SEP-2640 limits | high |

A critical or high finding quarantines the version. Static rules miss
natural-language payloads; the verification gate, scope confinement of
learned skills and approval of promotions are the other layers.

**Permissions.** Declared in `metadata` as `kleos.permissions: "network,
shell"` (metadata values are strings per the spec); recorded in the
version's provenance; `allowed-tools` is recorded, never honoured for skills
served over MCP (SEP-2640 host rule).

**Execution policy.** kleos core stores and serves scripts as content; it
never executes them (manifest entries of `scripts/` carry `exec:
requires-approval`). A deployment that wants execution runs it outside kleos
or in a sandboxed `::flow` step of its own (e.g. a code-mode worker in a
container), after a human approves that version's digest.

**Digest pinning.** A version is its digest; SEP-2640 entries list every
file's digest and size, so hosts bind approvals to content; reads re-check
the stored bytes against their digest.

**Signatures** (`skill_sign.rs`): detached Ed25519 signatures in the
**minisign** format, offline (`ed25519-dalek`, `blake2`): prehashed mode
(`ED`, BLAKE2b-512 of the canonical manifest), a trusted comment
(`timestamp`, skill, digest) covered by the global signature. Public keys
are minisign keys, so `minisign -Vm manifest.json -P <key>` verifies a
package's manifest. Secret keys (`kleos-sk1:…`, from `keygen`) are passed per
call (`secret_key`, `key_file`, `key_env`) and never stored. `trust` /
`untrust` / `keys` manage the store's trusted keys; `sign`, `verify`.

## Sharing

**Promotion** `promote {name+scope | ver, to, approve?, actor?}`: to a
broader scope only (agent → team → workspace → org). An audited **copy**:
the target gets a new version with the same files and digest (`origin:
promoted`, provenance: source version, promotion, approver, requester); the
source is untouched; signatures carry over. Approval policy (aligned with the
tenancy decisions: skills are instance knowledge, so promotion is approved,
never automatic by default):

- a pending promotion is approved by an accountable human: `approve
  {promotion, actor: {human}}` (`reject` declines);
- `approve: true` on the request with an `actor.human` applies it at once,
  except into the org scope, which always needs a separate `approve`;
- `config {policy: {auto_promote_to: ['team'], auto_min_uses,
  auto_min_success}}` allows automatic promotion into listed kinds (never
  org) for versions with enough uses and success, no scripts and a clean scan.

Under tenancy, `approve` will check the actor's `promote` / write right on the
target (`::tenant approve` semantics); the data model already records both
actor chains.

**Packages** `package {…, secret_key?}` → `{format: "kleos-skill-package/1",
name, description, version, digest, manifest, files (base64), signatures:
[{kid, public_key, signature}], provenance, stats, scan}`. **Registry**:
`publish` stores a package in this store's registry (`kleos_skill_pub`),
served by `kleos server` at `GET /skills/registry?query=` and `GET
/skills/registry/:name`; `registry {query? | name?}` reads it. **Install**
`install {package | path | url + name, scope?, require_signature? (true),
activate?}`: decodes the files, checks each against the manifest and the
manifest against the digest, requires a valid signature from a trusted key
(policy `require_signature`), runs the scanner, stores the version
(`origin: installed`) with its signatures, and activates it only if asked and
clean.

## MCP: skills over MCP (SEP-2640, Final)

The kleos MCP server declares `capabilities.extensions
["io.modelcontextprotocol/skills"] = {"directoryRead": true}` (with
`resources`):

- `skills/list {cursor?}` → `{skills: [{uri, frontmatter, resources: [{uri,
  digest, size}], _meta}], nextCursor?, ttlMs, cacheScope}`. A **partial
  listing** by default (the extension allows it): the 32 most useful live
  skills (posterior × usage). `kleos mcp --skills-list full` pages the whole
  catalog; `none` lists nothing (hosts use `find_skills` and `skills/get`).
  kleos extension: `{task, k}` returns the top-k for a task.
- `skills/get {uri}` → `{skill: entry}` for any live skill, listed or not;
  unknown → `-32602`.
- `resources/read` on `skill://…` serves the file (verified against its
  digest); `resources/directory/read {uri}` lists a directory's children
  (`inode/directory` for subdirectories).
- URIs: `skill://<scope prefix>/<name>/<path>`; the default scope has no
  prefix (`skill://kleos-memory/SKILL.md`), `team:billing` →
  `skill://team/billing/refunds/SKILL.md`. The last segment before the file
  is always the skill name. The frontmatter in an entry is parsed from the
  stored `SKILL.md`, so it is identical to the file a host fetches.
- Tools: `find_skills` (read-only), `get_skill` (read-only),
  `record_skill_outcome`, `propose_skill` (never activates), `skill_admin`
  (`history`, `stats`, `activate`, `rollback`, `undo`, `retire` — the last four
  confirm through MRTR or `confirm: true` — `promote`, `learn`, `optimize`).

## CLI

`kleos skill <op> [JSON] [--name N] [--scope S] [--ver V] [--task T]
[--path P] [-k N] [--budget N] [--activate] [--wait] [--yes] [--json]`, for
every op (`outcome` = `record_outcome`, `trace` = `record_trace`). Human
output for `find`, `list`, `import`; `rollback`, `undo`, `retire`, `erase`,
`release` ask for confirmation (`--yes`).

## Ops

| Group | Ops |
|---|---|
| setup | `init {embedder?}`, `config {embedder?, lifecycle?, policy?, learn?, router?}` |
| content | `put`, `import`, `export`, `get {…, file?, files?}`, `list`, `history`, `as_of` |
| serving | `find`, `entry`, `entries`, `read`, `dir` |
| lifecycle | `activate`, `rollback`, `undo`, `retire`, `quarantine`, `release`, `erase`, `lifecycle`, `scan` |
| outcomes | `record_outcome`, `use`, `stats`, `record_trace`, `traces`, `eval_add`, `evals`, `eval` |
| learning | `learn`, `optimize`, `proposals`, `propose` (`learn_stage`, `optimize_stage` are the workflow steps) |
| sharing | `promote`, `approve`, `reject`, `promotions`, `keygen`, `trust`, `untrust`, `keys`, `sign`, `verify`, `package`, `publish`, `registry`, `install` |

Read-only callers (`::skill` in an immutable script, read-only HTTP
tokens, read-only MCP connections) may call the read ops; `find` then does
not log its route.

## Running the workflows

`skill.learn` and `skill.optimize` run on the flow queue `skills`. A `learn`
or `optimize` call through `DbInstance` (every binding, HTTP, MCP, CLI, a
`::skill` script, a principal op in any partition) starts an in-process
runner thread (`DbInstance::skill_kick`) that ticks the
queue with a one-hour lease (their steps replay tasks through an LLM for
minutes) until nothing is due. Scheduled runs are fired by any flow worker
that fires schedules (`kleos server --flow-worker`, `DbInstance::flow_worker`);
give such a worker the `skills` queue and a long lease.

## Tenancy

As a principal (`::principal skill {as, op, …}`, `memory_call_as(ctx,
"skill", {op, …})`, and every MCP skill method and tool on a principal
connection), the skill library follows docs/specs/tenancy.md
(`kleos-core/src/tenancy_skill.rs`):

- **Partitions.** Skills live in the workspace's partition; org-scope skills
  in the org partition. Skill scopes are tenancy namespaces (`ws`,
  `team:<t>`, `user:<u>`, `au:<a>:<u>`, `run:…`, `org`).
- **Reads** (`find`, `entries` / `skills/list`, `list`, `get`, `entry` /
  `skills/get`, `read`, `dir`, `history`, `stats`, …) see only the skill
  scopes Cedar lets the context read; `find` and listings merge the
  workspace and (when readable) the org partition; anything else is "not
  found".
- **Writes** need `write` on the skill's scope (`put`, `propose`, `activate`,
  `rollback`, `undo`, `retire`, `optimize`, `sign`, `publish`, `install`
  with a package, …); a new skill's scope defaults to the narrowest sensible
  one (as `remember`); outcomes, `use`, traces and `eval` need `read`.
- **Promotion** needs `promote` on the target; into `org` it always waits
  for a human with the `approve` right (an org admin), however it was
  requested; `approve` / `reject` need a human in the context and `approve`
  on the target. The store's auto-promotion policy does not apply to
  principals.
- **Actors and audit.** `actor` is `{human, agent, run, chain}` from the
  context on every version, head change, outcome, trace and log row; every
  op is appended to the workspace's hash-chained audit (`skill.<op>`,
  allow / deny / pending) and the data writes go through the session's
  audit sink.
- **Refused to principals:** `path` and `url` inputs (server files and
  network), and store administration (`init`, `config`, `import`,
  `export`, `erase`, `lifecycle`, keys).
- A store without tenancy behaves as before.

## Limits

- One embedder per store's skill index (changing it re-embeds the index).
- Verification replays tasks with an LLM as the agent; skills whose value
  shows only with tools need a code-mode verifier (record outcomes from a
  real harness with `eval`-style cases, or report outcomes and let the
  Bayesian rollback judge).
- The scanner is rules only (no classifier); natural-language payloads can
  pass it.
- `install {url}` needs the `requests` feature.
