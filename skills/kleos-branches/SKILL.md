---
name: kleos-branches
description: Experiment safely on a copy of kleos memory with branches (fork, diff, merge, rewind, discard). Use before risky or speculative memory changes, when trying alternatives in parallel, or when the user wants to undo what was learned in a session.
---

# kleos branches

A fork is an instant copy of a namespace's memory; changes in the fork never
touch the parent until merged. Merging replays the fork's statements, so
kleos resolves conflicts by meaning (a newer fact replaces the older one).

## Workflow

```bash
kleos branch fork acme-try --from acme          # instant copy
kleos remember "Assume the launch moves to May." --ns acme-try
kleos branch diff acme acme-try                 # what differs
kleos branch merge acme-try --into acme --dry-run   # exact preview, always first
kleos branch merge acme-try --into acme --yes       # apply after the user agrees
kleos branch discard acme-try --yes             # or throw it away
```

MCP: `branch {action: "fork" | "diff" | "merge" | "rewind" | "discard" |
"list", …}`; merge (not dry-run), rewind and discard ask for confirmation.

## Undo a session

`kleos branch rewind acme <timestamp-µs> --yes` puts memory back as it was at
that time. It is undoable (the result includes an `undo` time). Use it when
the user says "forget everything from this session" and confirm first.

## Rules of thumb

- Always `--dry-run` a merge and show the user what will change.
- One branch per experiment or per parallel agent; discard the losers.
- Don't fork for ordinary remembering — only for speculative changes.
