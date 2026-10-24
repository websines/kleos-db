---
name: kleos-ontology
description: Inspect and correct what kleos has learned about the structure of a memory store (its ontology - kinds of facts, which ones hold one value at a time). Use when the user asks why a fact replaced (or did not replace) another, what kleos has learned, or wants an automatic ontology change undone.
---

# kleos ontology

kleos types facts (lives_in, employer, likes, …) with an ontology that is on
by default and maintains itself: it learns new properties from memories and
documents, merges synonyms and corrects whether a property holds one value
(a new value replaces the old) or many (values accumulate). Every change is
recorded and can be undone. You normally don't need to touch it.

## Look

```bash
kleos ontology list                 # ontologies and which namespaces use them
kleos ontology history --ns acme    # changes it made to itself, with evidence
```

Use these to explain behaviour: e.g. "employer holds one value, so 'joined
Stripe' replaced 'works at Google'", or "likes holds many values, so both
stay current".

## Fix

- Undo one automatic change: `kleos ontology undo <change-id> --ns acme`.
- Learn now (it also runs on its own): `kleos ontology learn --ns acme`.

Undo only when the user agrees the change is wrong (e.g. a property wrongly
made single-valued is replacing facts that should coexist). There are no
MCP tools for the ontology; use the CLI, or the read-only `query` tool if
enabled.
