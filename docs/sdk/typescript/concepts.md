# Concepts

What kleos does with what you give it, in plain words. The full detail is in
the specs under [`docs/specs/`](../../specs/); this page is the short
version.

## Memories and claims

A **memory** is one stored statement with a time it became true (`valid_from`,
default: when you said it). `remember(text)` splits a text of several
sentences into **claims**, one memory each:

> "I moved to Lisbon in March. My sister Anna lives in Munich."
> → two memories, `m_….0` and `m_….1`, both linked to the whole text.

Recall shows the whole statement around a claim, so context isn't lost
(`split: 'none'` stores the text as one memory). Results report the claims in
`claims`.

## Supersession and pending

When a new fact replaces an old one ("I moved to Lisbon" after "I live in
Paris"), kleos **supersedes** the old memory: it stops being current but is
never deleted, so you can still read it in the past. The decider model makes
this call. It only answers when it's confident:

- **auto:** the old memory is superseded at once (`superseded` in the result);
- **pending:** the decision waits for review (`tentative` in the result). The
  older memory shows `pending_superseded_by`, and `recall(…, { hidePending:
  true })` leaves it out;
- **none:** both stay current.

Exact restatements are **duplicates**: nothing new is stored, and the source is
added to the existing memory.

kleos also keeps impossible replacements out. Small talk never replaces
anything, facts about different people never replace each other, and
many-valued facts ("likes", "visited") never supersede (see the ontology,
below). Without a decider, every memory is kept as a plain insert.

`reflect(ns)` lists what is pending, possible contradictions and recent
changes.

## Namespaces

`ns` keeps memories apart: one per user, team, project or agent. Recall only
sees its own namespace. Namespaces cost nothing to create; just use a new name.
The default is `default`.

## Time travel

Every change is kept with valid time. `recall(query, { asOf: t })` reads memory
as it was at `t` (microseconds since the epoch: `Date.parse('2026-03-01') *
1000`), so you can answer "what did the agent know when it decided?".
`forget` retracts by default (the memory stops being current, history stays);
`mode: 'erase'` deletes every version, for GDPR requests.

## Documents with citations

`remember(text, { attachments: [...] })` stores documents alongside the
statement: PDF (text layer or OCR), images, Markdown, HTML, DOCX.

- **Chunks:** each document is split into chunks that keep its headings, and
  each chunk is a memory with its page and section.
- **Citations:** recall searches documents and memories together, and context
  lines cite their source: `[handbook.md, p. 4, § Refunds]`.
- **Small documents** (up to 4 pages) are ready when `remember` returns.
  **Big ones** come back with `status: 'running'` and ingest in the
  background, crash-safe, becoming searchable batch by batch.
- **Re-uploading** the same bytes does nothing; a changed file re-embeds only
  its changed chunks.

## Entities and the ontology

An **entity** is someone or something statements are about: `user`, "Anna",
"Acme Corp".

- **Resolution:** kleos resolves names to entities, so "Anna" and "my sister
  Anna" become one node, and keeps a profile of each (`profile(name)`).
- **The ontology** describes what kinds of facts exist: `lives_in` holds one
  value at a time, `likes` many, `sister_of` is a family link. It's what tells
  kleos which facts can replace each other.
- **Zero maintenance:** a personal ontology is on by default. kleos learns new
  fact types from your data, fixes its own mistakes (a type that keeps getting
  two values becomes many-valued), and every change is versioned and undoable
  (`ontology('history')`, `ontology('undo')`). You never have to write or
  maintain one; importing your own (OWL/Turtle or JSON) is optional.

## Branches

`fork(ns, 'try')` makes a branch of a namespace instantly (copy-on-write). The
branch sees its parent's memories as of the fork and keeps its own changes
separate. `diff`, `merge` (replays the branch's statements through `remember`,
so conflicts are settled the normal way), `rewind` (undoable) and `discard`
finish the job. Use them to try something in an agent run and keep only what
worked.

## Durable workflows

A workflow is a set of steps that survives crashes and long waits. Steps can
query, remember, recall, sleep, or wait for a signal (a human approval days
later). Each step is saved as it finishes, so after a restart the workflow
continues where it stopped, and a step's memory writes happen exactly once.
Background document ingestion is a workflow too. Embedded stores run them with
`flowWorker()`; servers with `--flow-worker`.

## Priority lanes

Model work goes through two lanes:

- **interactive** (the default): recall and chat messages go first;
- **bulk**: document pages and trace batches use the gaps, and are guaranteed
  a share so they never starve.

A big PDF ingesting in the background doesn't slow an agent's recall. Pass
`priority: 'bulk'` to `remember` or `recall` for background work. All models
share one memory budget (`KLEOS_MEMORY_BUDGET`), so memory stays flat however
much work is queued.
