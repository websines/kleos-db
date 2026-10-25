---
name: kleos-memory
description: Long-term memory with kleos. Use when answering anything that may depend on earlier conversations, the user's preferences or facts, project decisions or history; and when the user states something worth keeping (a fact about themselves or the project, a decision, a preference, a correction). Uses the `kleos` CLI or the kleos MCP tools.
---

# kleos memory

kleos stores statements, splits them into facts, dates them, links them to
people and things, and keeps history: when a fact changes, the old value is
kept as "previously". Use one namespace per user or project (`--ns`).

## Before answering: recall

Recall when the answer could depend on what was said before (preferences,
names, decisions, deadlines, "what did we decide about…").

```bash
kleos recall "what database did we choose for billing" --ns acme
```

MCP: `recall {query, ns}`. The default output is a **context block**:

- grouped by entity, oldest first, each line dated;
- `previously: X (from … until …)` — an older value that was replaced; use
  the current one, mention the change if relevant;
- `possibly replaced by` / pending — an uncertain replacement; say it is
  uncertain or ask the user rather than picking one;
- `[file.pdf p. 4, § Results]` — a document citation; cite it.

Keep it small: `--budget 600` (tokens) for routine questions, more for broad
ones. Use `--list` (MCP `format: "list"`) only when you need memory ids.
Do not recall for general-knowledge questions.

## When the user says something worth keeping: remember

Remember stable facts, preferences, decisions, commitments and corrections —
one clear statement each, in the user's words where possible:

```bash
kleos remember "We switched the billing service from MySQL to Postgres." --ns acme
```

MCP: `remember {text, ns}`. kleos detects that this replaces the old fact; do
not forget the old one yourself. Don't remember chit-chat, questions,
your own guesses, or secrets (passwords, keys, card numbers).

## Review and forget

- `kleos reflect --ns acme` (MCP `reflect`) lists pending replacements and
  possible contradictions — useful before a summary or when the user asks
  what you know.
- `kleos profile Anna --ns acme` (MCP `entity_profile`) — everything known
  about one person or thing.
- Forget only when the user asks: `kleos forget <id> --ns acme --yes`
  (retract, keeps history) or `--erase` (delete everywhere, e.g. privacy
  requests). Confirm with the user first; the MCP `forget` tool asks for
  confirmation itself.

## Setup

`--db PATH` for a local store or `--url URL --token T` for a kleos server; or
put them in `~/.config/kleos/config.json` (`{"db": "...", "ns": "..."}`).
Add `--json` for machine-readable output.

## In a shared deployment (principal mode)

When kleos runs with tenancy (docs/specs/tenancy.md), your token says who you
are and whom you act for; `ns` is ignored. Memory you store goes to your
private scope with the user (`agent_user`) unless you pass `scope`
(`team:<id>`, `workspace`). Do not move something private into a shared
scope on your own: ask the user, then `promote` it (a human may have to
approve). Recall only returns what everyone in the conversation may see, and
each hit says its `scope`.
