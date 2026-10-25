# Tenancy, identity and access control

Status: implemented (wave7/tenancy, 2026-10-25; follow-up closing the first cut's deferred items the same day). "Limits" lists what is still not built and why.

Design basis: [research/tenancy-research.md](../../../research/tenancy-research.md) (in the parent repo), its "Recommendations", "(h) v1 versus later" and "Flags". The decisions below are final for v1.

Code:
- `kleos-core/src/runtime/transact.rs`, `relation.rs`, `db.rs`: workspace **partitions** (relation-name routing, `Db::partition_view`), the principal view state (`ViewCtx`) and the commit-time write sink;
- `kleos-core/src/tenancy.rs`: the model, admin ops (`tenant_call`), `PrincipalContext` and actor chains, scope keys, the hash-chained audit log with pseudonyms, erasure;
- `kleos-core/src/tenancy_authz.rs`: the Cedar policies, the projected entity graph, the intersection rule and the cached `ScopeSet`;
- `kleos-core/src/tenancy_mem.rs`: memory ops as a principal (`memory_call_as`), recall across scopes, promotion and approvals, filtered subscriptions;
- `kleos-core/src/memory_branch.rs` (`Scope::union`), `memory.rs` (the union pre-filter, read-time precedence, per-scope profile composition, `kleos_mem_actor` rows, schema v9, the erase cascade to graph attributes), `ontology.rs` / `ontology_learn.rs` (the workspace learned layer);
- surfaces: `::tenant` / `::principal` (run by `DbInstance`), `kleos-bin/src/auth.rs` (JWT, API keys, RFC 9728), `server.rs` (`--tenancy`), `mcp.rs` (tools as a principal), `cli.rs` (`kleos tenant`, `--as-human`, `--as-agent`), `kleos-lib-python`, `kleos-lib-nodejs`, `sdk/ts`.

Tests: `kleos-core/tests/tenancy.rs` (11 tests plus an ignored perf test), `kleos-bin/src/auth.rs` (2), `kleos-bin/tests/mcp.rs::principal_mode_tools`.

## Goal

One kleos store serves many organisations, each with workspaces, teams, people and agents. Humans and agents are identified at every level: both are principals, both can be members of an org, a workspace or a team, hold roles and own memory. Every request carries the chain of actors behind it, every write and every read records it, and every agent has an accountable human.

A store with no tenancy configured behaves exactly as before: one implicit org and workspace, an implicit full-access caller, the same relations, queries and costs.

## Model

### Hierarchy

```
Org ──< Workspace           (data and index boundary)
 │
 ├──< Team (tree, depth ≤ 4) ──granted roles on──> Workspace (many-to-many)
 └──< Principal: human | agent (sponsor: human, NOT NULL) | service
```

- **Org**: identity, policy root, the unit of erasure keys and of the org scope. Principals belong to orgs through an org membership with a role: `owner`, `admin` or `member`.
- **Workspace**: belongs to one org. It is the **physical partition**: its memories, vectors, FTS index, ontology graph, derived data, flows and documents live in their own relations (see "Partitions"). Org-scope data lives in the org's own partition.
- **Team**: defined at org level, with an optional parent team (a tree, depth ≤ 4). A member of a child team inherits the parent's access (GitHub semantics). Teams are granted roles on workspaces.
- **Principal**: a store-wide id (`[A-Za-z0-9._-]{1,64}`) with a kind:
  - `human`;
  - `agent`: has a mandatory `sponsor` (a human of the same org, accountable for it) and an optional `template` (another agent principal acting as its definition: instances of one template share their per-user memory);
  - `service`: a non-agent machine identity (sync jobs, importers).
  Principals may carry external identities `(iss, sub)` for token mapping.

### Roles and grants

- Workspace roles: `reader` < `writer` < `admin`. A grant gives a role on a workspace to a principal or to a team (and so to the team's subtree). Extra actions can be attached to a grant: `promote` (agents may promote instance facts without approval).
- Org roles: `member` (may receive grants, reads org scope), `admin` and `owner` (manage teams, principals, workspaces, policies; approve promotions into org scope). Org admins do not get workspace data access implicitly.
- Changing any membership, grant, policy or principal bumps the **membership epoch**, which invalidates every cached `ScopeSet` at once.

### Scopes

A scope is a typed tuple. Inside a partition it is stored as the memory **namespace** (`ns`), so every existing per-namespace guarantee (supersession and duplicates only within the namespace, per-namespace entities, ontology facts, profiles and summaries) becomes a per-scope guarantee without new code.

| Kind | Tuple | `ns` | Partition | Default writer | Readers |
|---|---|---|---|---|---|
| `run` | (agent, user, run) | `run:<agent>:<user\|->:<run>` | workspace | the run's agent | the agent, the user |
| `session` | (agent, run, members) | `sess:<agent\|->:<run>:<m1>+<m2>…` | workspace | **default target of a shared session** (a context with participants) | its members and its agent: an ad-hoc intersection scope |
| `agent_user` | (agent, user) | `au:<agent>:<user>` | workspace | **default target of an agent acting for a human** | the agent, the user |
| `user` | (user) | `user:<user>` | workspace | default target of a human alone | the user; agents acting for the user (link assurance ≥ `issuer_endorsed`) |
| `agent` | (agent) | `agent:<agent>` | workspace | promotion only | workspace readers |
| `team` | (team) | `team:<team>` | workspace | writers in the team subtree | readers in the team subtree |
| `workspace` | (ws) | `ws` | workspace | writers | readers |
| `org` | (org) | `org` | org | promotion only (approved) | org members |

`<agent>` is the agent's template when it has one. Namespaces of the implicit `default` workspace that predate tenancy are **legacy scopes**: any other `ns` string is read as kind `ns`, with workspace readers and writers (this is how existing namespaces map into the implicit workspace).

Precedence, narrowest first: `run` / `session` > `agent_user` > `user` > `agent` > `team` > `workspace` / `ns` > `org`.

A **session** scope's members are the context's human and participants (sorted). A later context reads it only when every one of its actors is a member (or its agent), so a shared conversation's memory is visible to the same group, to each member alone, and to nobody who was not there.

### Principal context and actor chain

Every principal-mode call carries a `PrincipalContext`:

```rust
pub struct PrincipalContext {
    pub workspace: Option<String>,   // default "default"
    pub human: Option<String>,       // originating human principal
    pub agent: Option<String>,       // acting agent principal
    pub run: Option<String>,         // run / session id
    pub participants: Vec<String>,   // other principals in a shared session
    pub chain: Vec<ActorLink>,       // innermost (originator) first; derived when empty
    pub token: Option<TokenGrant>,   // what the credential allows (actions, scope kinds)
    pub purpose: Option<String>, pub txn: Option<String>, pub client_id: Option<String>,
}
pub struct ActorLink { kind: Kind /* human | ai_agent | service */, iss: Option<String>, sub: String,
                       principal: Option<String>, assurance: Assurance }
pub enum Assurance { Verified, IssuerEndorsed, HostAsserted, SelfAsserted }
```

- **Embedded mode**: the host builds the context; links are `host_asserted`.
- **Server / MCP mode**: the context is derived from the validated credential (see "Surfaces"); the token subject is `verified`, `act` links are `issuer_endorsed`, headers without proof are `self_asserted`.
- The accountable human is the originating human, or the agent's sponsor for an autonomous run (no human).
- Validation: every named principal must exist, be active, and belong to the workspace's org; an agent's template and sponsor must be valid; the chain depth is at most 6 (deeper is an error, never truncated).

## Authorization

### Effective permissions (the intersection rule)

```
allowed(ctx, action, scope) =  token_allows(action, scope.kind)
                            ∧  ∀ actor ∈ {human, agent, participants…, sponsor if ceiling}: cedar(actor, action, scope, ctx)
```

- human ∩ agent ∩ token ∩ every participant of a shared session (MSR 2509.14608). A participant can only take access away.
- An **autonomous** agent (no human) runs on its own grants. The org policy `sponsor_ceiling: true` adds its sponsor to the intersection.
- Agent-owned scopes of an autonomous agent: `run(A, -, *)`, `agent(A)`; never `agent_user(A, *)` (that would read across users).

### Cedar

kleos embeds `cedar-policy` (4.x) for the scope-level decision. The entity graph is projected from the tenancy relations per request:

- `Kleos::Principal::"<id>"` `{kind, sponsor?}`, parents: its teams, its org-member group, its direct workspace role groups;
- `Kleos::Team::"<id>"`, parents: its parent team and its workspace role groups;
- `Kleos::Group::"<ws>/reader|writer|admin|promote"` (admin ∈ writer ∈ reader), `Kleos::Group::"org:<org>/member|admin"`;
- `Kleos::Scope::"<ws>|<ns>"` `{kind, owners: Set<Principal>, team?, ws_reader, ws_writer, ws_promote, org_member, org_admin}`.

The default policy set (`tenancy_authz.rs`, `DEFAULT_POLICIES`) encodes the reader/writer columns of the scope table, plus:

- `promote`: humans who can write the target; agents in the target workspace's `promote` group;
- user scope reads by an agent need `context.human_assurance` ≥ `issuer_endorsed` (an API key with a self-asserted on-behalf-of header cannot read a person's own memory).

Orgs add policies (`::tenant policy_set {org, cedar}`), typically `forbid` rules, e.g.:

```
forbid(principal, action == Kleos::Action::"read", resource)
  when { principal.kind == "agent" && resource.kind == "team" };
```

Cedar is evaluated over a handful of **candidate scopes** per request (the actor's personal scopes, its teams, the workspace, agent scopes, legacy namespaces, the org), never per memory.

### The ScopeSet

`scope_set(ctx, action)` → the namespaces of the partition (and whether the org scope is included) that the context may read or write. Computed once per request, cached by `(context hash, membership epoch, scope epoch)`; the scope epoch moves when a new scope namespace is created. Only scopes that exist (have a namespace row) are returned for reads.

**Why not Cedar partial evaluation (TPE).** `cedar-policy`'s `tpe` (`query_resource`) would enumerate the scopes a principal may read from residual policies instead of evaluating concrete candidates. It is not used: the index pre-filter needs the concrete namespaces of the partition anyway, so enumeration over the partition's namespace list is unavoidable; the candidates are pre-filtered to the actors' own personal and session scopes, their teams, the workspace and agent scopes (a handful per request, whatever the number of namespaces), and Cedar is evaluated over those only. Measured (release, see "Results"): an uncached session plus ScopeSet costs 0.7 ms, a cached one 0.3 ms. TPE is also an experimental API (feature `tpe`, subject to change). If org policies grow beyond what concrete evaluation handles, TPE can replace the per-candidate loop in `tenancy_authz.rs` without changing callers.

### Enforcement points

1. **Routing (hard wall).** A principal call runs on the workspace's partition view; no relation of another workspace can be named from it.
2. **Pre-filter in the indexes.** Recall searches with `filter: is_in(ns, $scopes) && current, prefixes: $scopes` in HNSW (the filter runs inside the bottom-layer search; for very selective filters the exact scan takes over and reads **only the scopes' key ranges** — the `prefixes` option) and `is_in(ns, $scopes), prefixes: $scopes` in FTS, where BM25's `N` and every document frequency are counted over the scopes' key ranges only, so scores reveal nothing about other scopes. Never a post-filter of a fixed top-k.
3. **Graph walk.** The PPR walk, inferred-link expansion and entity expansion read only rows of the scope namespaces (facts, mentions, types, entities), so invisible nodes are never loaded: no ranking mass and no `via` explanation can pass through them.
4. **Derived data.** Profiles and summaries are built per scope namespace (inputs from that scope only) and composed at read time by precedence.
5. **Writes** check `write` on the target scope; promotion checks `read` on the source and `promote` on the target.
6. **Errors and counts.** An id outside the ScopeSet is "not found", never "forbidden"; results never count invisible rows.
7. **Raw CozoScript, export, backup** are admin-only in server mode (as before); `::principal` ops are the principal surface.

## Partitions (physical isolation)

`Db::partition_view(name)` returns a view of the database in which every stored relation name is routed to `<name>@<relation>` (temp relations `_…` and the global control plane `kleos_tn_*` excepted). A view behaves like a separate database: `::relations` lists only its own relations (without the prefix), `:create` creates in it, indexes (`rel:idx`) follow their base relation, `::remove` drops only its relations. HNSW graphs, FTS postings and BM25 statistics, ontology graphs, flows and documents are therefore physically separate per workspace; a crossing needs a routing bug, not a filter bug. Dropping a workspace drops its relations.

- Workspace `w` → partition `ws_<w>`; org `o` → `org_<o>`.
- The implicit `default` workspace is the unpartitioned store, so existing data stays where it is.
- The root view hides partitioned relations from `::relations`.
- Each partition has its own runtime state: access-count buffer, document runner and consolidation runner. `DbInstance::flow_worker` (and so `kleos server --flow-worker`) serves the plain store and every workspace and org partition (re-listed every few seconds), so workflows, document ingestions and Engram workflows of a workspace run without a hand-made partition view; a principal call starts its partitions' document and consolidation runners as `remember` does for the plain store.
- Cached embeddings are keyed by the caller's workspace and ScopeSet as well as the text (`embed::with_cache_scope`), so cache timing reveals nothing across scopes.

## Schema

Control plane (global, created by the first `::tenant` op):

| Relation | Columns |
|---|---|
| `kleos_tn_org` | `{org => name, policy: Json, created, status}` (`policy`: `{sponsor_ceiling, cedar}`) |
| `kleos_tn_ws` | `{ws => org, name, partition, created, status}` |
| `kleos_tn_team` | `{team => org, name, parent?, created}` |
| `kleos_tn_principal` | `{id => org, kind, name, sponsor?, template?, status, created}` |
| `kleos_tn_ident` | `{iss, sub => principal}` (external identities) |
| `kleos_tn_member` | `{group, principal => role, created}` (`group`: `org:<org>` or `team:<team>`) |
| `kleos_tn_grant` | `{ws, grantee => role, actions: [String], created}` (`grantee`: `p:<id>` or `t:<team>`) |
| `kleos_tn_apikey` | `{hash => principal, actions: [String], label, created, expires?}` (SHA-256 of the key) |
| `kleos_tn_key` | `{principal => key: Bytes, created}` (pseudonym keys; deleted on erasure) |
| `kleos_tn_promotion` | `{id => ws, from_ns, ids: [String], to, chain: Json, status, approved_by?, at}` |
| `kleos_tn_audit` | `{ws, seq => ts, op, chain: Json, accountable, targets: Json, relations: [String], scope_set, decision: Json, otel: Json, prev, hash}` |
| `kleos_tn_meta` | `{key => value}` (`epoch`, `scope_epoch:<ws>`, `version`) |
| `kleos_tn_audit_head` | `{ws => seq, hash}` (the chain head, written with each event) |
| `kleos_tn_request` | `{id => ws, op, ns, ids: [String], args: Json, chain: Json, requested_by, status, approved_by?, result?, at}` (curation requests) |
| `kleos_tn_jwks` | `{principal => jwks: Json, updated}` (keys for `private_key_jwt`) |

Memory schema **v9** (every partition, and the root store on its next open; created empty):

| Relation | Columns | Notes |
|---|---|---|
| `kleos_mem_actor` | `{ns, id => human?, agent?, run?, chain: Json, at}` | who wrote a memory, written in the memory's own transaction |
| `kleos_mem_promoted` | `{ns, id => from_ws, from_ns, from_id, chain: Json, approved_by?, at}` | provenance of a promoted copy |

A v8 store gains both on first use; nothing is backfilled, and nothing is written to them outside principal mode.

## Operations

### Admin (`::tenant <op> {…}`, `Db::tenant_call(op, args)`)

| Op | Args | Notes |
|---|---|---|
| `org_create` | `org, name?, owner?` | `owner` (a human id) is created and made `owner` |
| `ws_create` | `ws, org, name?` | creates the partition and initialises its memory store with the root store's embedder and decider |
| `team_create` | `team, org, parent?` | depth ≤ 4 |
| `principal_create` | `id, org, kind, name?, sponsor?, template?, idents?: [[iss, sub]]` | `sponsor` required for agents |
| `member_add` / `member_remove` | `group: 'org:<o>' \| 'team:<t>', principal, role?` | |
| `grant` / `revoke` | `ws, principal \| team, role, actions?` | |
| `policy_set` | `org, cedar?, sponsor_ceiling?` | the Cedar text is validated before it is stored |
| `apikey_create` / `apikey_revoke` | `principal, actions?, label?` → `{key}` once | |
| `principal_key_set` | `id, jwks` | the public keys an agent signs `private_key_jwt` assertions with |
| `principal_disable` | `id` | |
| `erase` | `principal, reassign_to?` | see "Erasure" |
| `approve` / `reject` | `promotion \| request, as` | see "Promotion" and the curation of agent and org scopes |
| `audit` | `ws, from_seq?, limit?, principal?` | events; `principal` matches by pseudonym |
| `audit_verify` | `ws` | `{ok, events, broken_at?}` |
| `scopes` | `ctx, action?` | the ScopeSet, for debugging |
| `get` / `list` | `kind` | the model rows |

### As a principal (`::principal <op> {as_human: 'alice', as_workspace: 'eng', …}`, `Db::memory_call_as(ctx, op, args)`)

In CozoScript the context is either flat `as_*` fields (`as_workspace`, `as_human`, `as_agent`, `as_run`, `as_participants`) or `as: '<json>'` / `as: parse_json('…')`.

The memory ops take the same options as the memory API, plus:

- `remember {scope?}`: target scope: `run`, `agent_user`, `user`, `team:<id>`, `workspace`; default the narrowest sensible scope (agent + human → `agent_user`; human alone → `user`; autonomous agent → `run`; service → error unless given). `agent` and `org` are promotion targets only. Writes the memory's actor row and the audit events.
- `recall` / `context {scopes?}`: reads across the context's readable scopes (optionally narrowed by kind: `['agent_user', 'team', …]`); every hit has `scope`. Conflicts across scopes resolve by precedence: an exact restatement, or a value of the same subject and single-valued ontology property, in a broader scope is dropped in favour of the narrower one. Supersession never crosses scopes.
- `forget {ids | scope + filter}`, `set`, `link`: the target ids must be in readable scopes. In scopes the context writes they act directly; in **agent and org scope** a curator (whoever may promote into it: a writing human, the agent with the `promote` grant, an org admin for org scope) acts directly, and anyone else who reads the memory files a **request** (`pending: [{request, scope, ids}]`) that a human approves with `approve {request}` (or `::tenant approve {request, as}`).
- `query {script, params}` and `reembed {embedder}`: store-wide ops of the partition, for workspace admins; the audit keeps the query's SHA-256, not its text.
- `reflect {scope?}`, `consolidate {scope?}`: per scope.
- `promote {ids, to}`: see below.
- `fork | merge | discard {scope, branch}`: a branch of a scope is namespace `<scope ns>~<branch>` and inherits the scope's readers and writers.
- `subscribe {relation}` (Rust/`DbInstance` only: `subscribe_as`): a live subscription whose rows are filtered to the ScopeSet, re-evaluated per change.
- `flow | doc | ontology {op, scope?, …}`: the workflow, document and ontology ops of the partition; the namespace they address is a scope (default the workspace scope), read ops need read on it, the others write.
- `scopes {action?}`: the ScopeSet (debugging).

### Promotion

Promotion is a **copy with provenance**, never a move (declassification).

- The source ids must be readable by the context; the target needs `promote`.
- Humans who can write the target, and agents holding the `promote` action on the workspace, promote directly. Other agents create a **pending** promotion; a human with write on the target approves (`::tenant approve`).
- **Org scope** always needs an org admin's approval (the caller's own when it is an org admin human passing `approve: true`).
- The copy is a `remember` into the target scope (so the target's duplicate and supersession rules apply) plus a `kleos_mem_promoted` row; the audit records both chains.
- **Automatic promotion** is limited to schema-level knowledge: properties the zero-maintenance ontology learns in a scope are merged into the workspace's learned layer (`learned:ws`), which every scope of the workspace reads. Instance facts are never promoted automatically.

### Audit

Append-only, hash-chained per workspace (`kleos_tn_audit`):

- **Writes**: every transaction in a principal view that writes appends an event with the relations it wrote **inside that transaction**, just before it commits (the chain head `kleos_tn_audit_head` is read and written in the same transaction), so a write and its event commit together or not at all. This covers memories, claims, automatic ontology changes, supersession, forget, consolidation, documents, flows, branches and Engram jobs. Each op also appends an event with its target ids, the scope set hash and the decision.
- **Reads**: `recall`, `context` and `reflect` append an event with the returned ids (never content).
- `hash = SHA-256(prev ‖ canonical event)`; `audit_verify` recomputes the chain.
- Principal ids, `iss` and `sub` in events are **pseudonyms**: `HMAC-SHA-256(key[principal], id)`, with one random key per principal in `kleos_tn_key`. Erasure deletes the key, so the chain stays valid and the events become unlinkable (crypto-shredding).
- Each event carries GenAI-style attributes in `otel`: `gen_ai.operation.name`, `gen_ai.agent.id`, `gen_ai.conversation.id` (the run), `kleos.scope_set_hash`. With the feature `otel`, every audited op is also an OpenTelemetry span of the global tracer `kleos` (`<op> kleos`, the same attributes plus `kleos.decision` and `kleos.targets`); the host installs the tracer provider and exporter.

### Erasure

`::tenant erase {principal}` (a human or an agent):

1. Refused while the principal sponsors active agents, unless `reassign_to` names a new sponsor.
2. In every workspace partition of its org and the org partition: every memory of the scopes it owns (`user:<h>`, `au:*:<h>`, `run:*:<h>:*`; for an agent `au:<a>:*`, `run:<a>:*`, `agent:<a>`) and every memory it authored elsewhere (`kleos_mem_actor`) is erased with `forget {mode: 'erase'}`, which already cascades to vectors, sources, mentions, links, decisions, derived summaries and profiles citing them, and Engram jobs.
3. Ontology facts whose evidence was erased, memory types and actor/promotion rows of the erased ids are removed; the ScopeSet cache and the embedding cache are cleared.
4. Memberships, grants, identities and API keys are removed; the principal becomes `erased`; its pseudonym key is deleted.

## Surfaces

- **Rust**: `Db::tenant_call`, `Db::memory_call_as`, `Db::subscribe_as`, `Db::partition_view`, `Db::audit_verify`; `DbInstance` mirrors them (and runs `::tenant` / `::principal`).
- **CozoScript**: `::tenant <op> {…}`, `::principal <op> {as: {…}, …}`.
- **Python**: `db.tenant(op, args)`, `db.memory_as(ctx, op, args)`.
- **TS** (napi `tenantCall` / `memoryCallAs`; `sdk/ts`): `kleos.tenant(op, args)`, `kleos.memoryAs(ctx, op, args)` (remote: the server derives the principal from the token).
- **HTTP** (`kleos server`): with `--tenancy <config.json>`, `/memory/:op` and `/mcp` require a credential and run as that principal; `/tenant/:op` needs an org admin; `GET /.well-known/oauth-protected-resource` serves RFC 9728 metadata; 401s carry `WWW-Authenticate: Bearer resource_metadata="…"`.
  - **JWT access tokens** (RFC 9068) from configured issuers: keys from an inline JWKS, a PEM, a **JWKS URL** or **OIDC discovery** (fetched, cached for `jwks_ttl_secs` (300), refetched when a token names an unknown `kid` — key rotation — at most every 10 s); HS256 for tests. Checks: signature, `iss`, `aud` = the configured resource (RFC 8707), `exp`/`nbf`. `sub` → principal (via `kleos_tn_ident` or directly); `act` (nested, OAuth Actor Profile) → chain links with `sub_profile` (`user` | `ai_agent` | `service`); depth > 6 → `invalid_request`; an unknown presenting AI agent → 403 `actor_unauthorized`. The outermost `ai_agent` actor is the agent; the subject (when a `user`) is the human.
  - **SPIFFE JWT-SVIDs** for agent workloads: an issuer with `spiffe_trust_domain` verifies SVIDs (which may carry no `iss`) whose `sub` is `spiffe://<domain>/…`, mapped through `kleos_tn_ident` (`["spiffe://<domain>", "<spiffe id>"]`).
  - **Transaction Tokens** (`Txn-Token` header; issuers with `txn: true`, audience `txn_audience`): alone they are the credential; with an access token their subject must match. Their scope narrows the credential, `txn` goes to the audit, `req_wl` joins the chain as an `issuer_endorsed` service link.
  - **DPoP** (RFC 9449): a token with `cnf.jkt` needs a `DPoP` proof (asymmetric key, `typ: dpop+jwt`, matching `htm`/`htu`, fresh `iat`, `ath` of the token, unused `jti`) whose key thumbprint (RFC 7638) is the token's; `Authorization: DPoP …` and `Bearer` are both read.
  - **Token exchange** (RFC 8693, `POST /oauth/token`, on when `signing_secret_env` names a 32+ byte key; metadata at `/.well-known/oauth-authorization-server`): `subject_token` (an access token for this resource) plus an actor — `client_assertion` (`private_key_jwt`, signed with the agent's key registered by `::tenant principal_key_set {id, jwks}`), or `actor_token` (a JWT-SVID, an access token or kleos token of an agent, or an API key) — yields a kleos delegation token: `sub` the subject's principal, `act` the actor with the subject token's own `act` nested inside (inherited links kept, never dropped; depth ≤ 6), scopes only narrowed, `cnf.jkt` when the request carries a DPoP proof. kleos accepts its own tokens like any issuer's.
  - **API keys** (`kl_…`) for simple deployments: the key's principal (usually an agent) is `verified`; `X-Kleos-On-Behalf-Of` names a human as `self_asserted`.
  - Tokens are never forwarded downstream (MCP 2026-07-28).
- **MCP**: the same credential handling; every tool runs as the principal (`scope` / `scopes` arguments; `ns` is ignored): memory tools through `memory_call_as`, `doc_add` / `doc_status` / `flow` / `ontology` through the principal passthrough ops (documents are listed from every readable scope), `branch` on a scope's branches, `entity_profile` from the readable scopes, `consolidate` per scope (in the call, so the principal's chain is recorded), `reembed` and `query` for workspace admins. CIMD client ids are accepted as `client_id` and logged.
- **CLI**: `kleos tenant <op> '{json}'`; memory commands take `--as-human`, `--as-agent`, `--workspace`, `--run`, `--scope` (embedded store). Against a server, `--token kl_…` or a JWT is sent as a bearer token and the server derives the principal.

## Limits

- Precedence across scopes recognises exact restatements and same-subject values of single-valued ontology properties; other conflicts are shown side by side, each hit labelled with its `scope`.
- BM25 `avgdl` stays the partition's average document length (`N` and document frequencies, the counts, are per ScopeSet).
- Document ingestion steps run by the background runner are audited as the runner's writes (no principal chain); the `remember` that queued them carries the chain.
- Concurrent audit appends from several processes on one store conflict on the chain head like any write-write conflict (RocksDB and TiKV abort the later committer, which then retries as its caller does).
- Not built (needs infrastructure beyond an embedded store, or an adopted standard): an ID-JAG / XAA authorisation server, mTLS-bound tokens (the TLS terminator owns the client certificate), SPIFFE X.509-SVIDs, chain-integrity drafts and AAuth.

## Results

Deterministic tests (`cargo test -p kleos --features engram --test tenancy`, fake embedder and decider): workspace isolation and partitions, agent-user scopes across users and autonomous runs, the intersection with participants, credentials and an org `forbid` policy, shared-session scopes, the PPR walk and entity expansion inside visible scopes, a selective ScopeSet, per-scope summaries, promotion (pending, approval, provenance, agent grant, org with approval), curation of agent and org scopes with requests, supersession and duplicates within a scope and read-time precedence, per-ScopeSet BM25 and embedding cache, the flow worker serving a workspace partition, the erasure cascade with a verifying, shredded audit chain, audit coverage of writes and reads with tamper detection, admin queries and the CozoScript ops. `kleos-bin/src/auth.rs`: JWT actor chains, API keys, JWKS URL caching and rotation (a local HTTP server), JWT-SVIDs, Transaction Tokens, DPoP, token exchange with `private_key_jwt`. `kleos-bin/tests/mcp.rs::principal_mode_tools`: every tool family as a principal. `--features otel --lib otel`: the span export.

Perf sanity (`perf_scope_filter`, release build, M-series laptop, mem store, 4,000 memories of 64-d fake vectors, graph off, 50 queries): a principal recall over 3 of the workspace's 5 scopes (60% of its memories, one pre-filtered search) takes **41 ms** against **32 ms** for a plain one-namespace recall of a 4,000-memory store (first cut: 62 vs 43 ms on a busier machine); restricted to one scope it takes 34 ms (the partition's own one-namespace recall: 32 ms); the session plus cached ScopeSet costs 0.3 ms, uncached (after a membership change) 0.7 ms. The remaining gap is the union's filtered index legs (about 13 ms more than `ns == $ns`) and per-scope metadata reads. While measuring, two queries that scanned a whole namespace per id were reordered (the history read binds `id` first; a union reads pending decisions in one pass), which took the union from 150 ms to 62 ms.

## Threat model

| Leak path | Defence |
|---|---|
| Filter bug in a shared index crosses tenants | per-workspace partitions; the scope filter only works inside one |
| BM25 statistics reveal other tenants' or scopes' vocabulary | per-partition FTS indexes; within a workspace `N` and document frequencies are counted over the caller's ScopeSet |
| Low-selectivity filtered HNSW returns too few rows → post-filter temptation | in-loop filter, two-hop expansion, exact scan of the passing rows when cheaper |
| PPR mass or `via` through invisible nodes | the graph is loaded from visible scope namespaces only |
| Derived profiles and summaries quote other scopes | built per scope, composed at read time; erased with their sources |
| Embedding inversion | vectors live and die with their memory's scope and partition |
| Agent memory as a cross-user channel | default agent target is `agent_user`; `agent` scope only by promotion |
| Confused deputy (agent acting for a human) | intersection of human, agent, token and participants |
| Self-asserted "acting for Alice" | per-link assurance in the Cedar context; user scope needs ≥ `issuer_endorsed` |
| Stale permissions after revoke | membership epoch invalidates the ScopeSet cache immediately |
| Existence probing by id | invisible ids are "not found" |
| Live subscriptions deliver raw rows | principal subscriptions filter by ScopeSet |
| Shared caches (timing) | ScopeSet cache keyed by context; embedding cache keyed by workspace and ScopeSet |
| Multi-party session memory leaks to one participant's later, different company | session scopes: readable only by a subset of the session's members |
| A crash between a write and its audit event | the write event commits inside the write's transaction |
| Stolen bearer token replayed | DPoP-bound tokens need a fresh proof of the holder's key; Transaction Tokens bind a transaction id |
| Ontology learned schema leaks instance data | only property/type definitions are promoted automatically |
| Audit log as personal data | ids only, pseudonymised, crypto-shredded on erasure |
| Token passthrough | inbound tokens are never forwarded |

## v1 versus later

**v1**: everything above.

**Later**: an ID-JAG / XAA authorisation server; mTLS-bound tokens and X.509-SVIDs; chain-integrity drafts or AAuth if adopted; Cedar TPE if org policies outgrow concrete evaluation; a SpiceDB / OpenFGA adapter; per-memory ad-hoc sharing scopes; dedicated-store and dedicated-deployment tiers, BYOK; per-org quotas and fair-share scheduling of background work; filtered `avgdl`.
