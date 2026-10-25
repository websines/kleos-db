---
name: kleos-skills
description: Use and grow the kleos skill library (versioned, measured procedures). Use before a task that a known procedure may cover (find the few skills that fit instead of guessing), after following one (report whether it worked), and when a procedure worth keeping emerges (propose it). Uses the `kleos` CLI or the kleos MCP tools.
---

# kleos skills

kleos keeps procedures ("skills", the SKILL.md format) as versions with
measured success rates. It never hands you the whole library: ask for the
few that fit the task at hand.

## Before a task: find

```bash
kleos skill find --task "rotate the staging database credentials" -k 3
```

MCP: `find_skills {task, k?, budget?}`. Each hit has a fit score, its
success rate over real uses and a `skill://` URI; the answer carries a
`route` id. Load the one that fits (body only when you will follow it):

```bash
kleos skill get --name rotate-db-creds          # instructions + file list
kleos skill get --name rotate-db-creds '{"file": "references/checklist.md"}'
```

MCP: `get_skill {name | uri, file?}`. Scripts in a skill are content: kleos
never runs them, and neither should you without the user's approval.

## After: report the outcome

```bash
kleos skill outcome --name rotate-db-creds '{"success": true, "route": "sr_…"}'
```

MCP: `record_skill_outcome {name, success, route}`. Always pass the `route`
from the search: it trains the router, and a version that does worse than
its predecessor is rolled back automatically.

## Propose what worked

When you solved a recurring task in a way worth reusing, propose a focused
skill (name, one-sentence description saying when to use it, numbered
steps, pitfalls, a check). Add two or three test tasks with expected results
so it can be verified:

```bash
kleos skill propose '{"skill_md": "---\nname: …\ndescription: …\n---\n## Steps\n1. …", "eval": [{"task": "…", "expected": "…"}]}'
```

MCP: `propose_skill`. Proposals are scanned and stored as candidates; they
are never served until activated (by a verified learning run or a person).
You can also log a run transcript for the learning loop:
`kleos skill trace '{"task": "…", "transcript": "…", "output": "…", "success": true}'`.

## Lifecycle (with the user's consent)

- `kleos skill history --name X`, `kleos skill stats --name X`.
- `kleos skill activate --name X`, `rollback --name X`, `undo --name X`,
  `retire --name X` (MCP `skill_admin {action}`; asks for confirmation).
- `kleos skill promote --name X '{"to": "team:support"}'`: an audited copy to
  a broader scope; it waits for a human's approval.

## Rules of thumb

- Search with the concrete task, not a keyword; take the top hit only if it
  fits, otherwise proceed without a skill.
- Report every outcome you can judge, including failures.
- Never paste secrets, credentials or personal data into a proposed skill.
