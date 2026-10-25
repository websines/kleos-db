# kleos for TypeScript

kleos is a memory engine for AI agents. It stores what an agent learns
(statements, documents, entities), keeps track of which facts are current,
and gives back a short, dated context block for the next prompt. It runs
inside your process or as a server, with its own small local models; no LLM
is needed to store or recall.

The TypeScript SDK (`sdk/ts`, package `kleos`) is one API over two backends:

- **Embedded:** the store runs in your Node process through a native module.
  Fastest; no server to run.
- **Remote:** a pure-TypeScript HTTP client for a `kleos server`. Works in
  serverless functions, edge runtimes and browsers, where native modules
  can't load.

Switching is one line: `open({ path })` or `connect({ url })`.

## Contents

| Page | What it covers |
|---|---|
| [Getting started](getting-started.md) | Install, embedded vs remote, first `remember` and `recall`, models and the memory budget |
| [Concepts](concepts.md) | Memories and claims, supersession, namespaces, time travel, documents, entities and the ontology, branches, workflows, priority lanes |
| Guides | [Chat agent with per-user memory](guides/chat-agent.md) · [Documents and citations](guides/documents.md) · [Support agent with entity profiles](guides/support-profiles.md) · [Branches](guides/branches.md) · [Long-running workflows](guides/workflows.md) · [Skills](guides/skills.md) · [kleos over MCP](guides/mcp.md) · [Serverless and edge](guides/serverless.md) |
| [API reference](api.md) | Every method, its options and results |
| [Python equivalents](python.md) | The same calls in the Python binding |
| [Troubleshooting](troubleshooting.md) | Native builds, model downloads, the memory budget, errors |

Every guide is tied to a script under [`examples/ts/`](../../../examples/ts/);
the SDK's tests run them all and check that the code shown in these pages
matches them.

## Status

The package is not published to npm yet. Build it from the repository (see
[Getting started](getting-started.md)). Node 18 or later; the examples use
TypeScript directly and need Node 22.6 or later.
