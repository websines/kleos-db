#!/usr/bin/env python3
"""Context cost of kleos's agent surfaces (docs/specs/mcp-cli-skills.md).

An MCP client puts every tool definition in the model's context on every
turn; a skill puts only its name and description there until the model opens
it. Token counts are estimated as characters / 4 (the estimate kleos uses for
its own context budgets), so read them as relative sizes.

    python3 scripts/mcp/token_cost.py <path-to-kleos-binary> <repo-root>
"""
import glob, json, os, subprocess, sys

BIN, ROOT = sys.argv[1], sys.argv[2]
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}}


def tok(s):
    return (len(s) + 3) // 4


def tools(extra):
    p = subprocess.run([BIN, "mcp", "--engine", "mem", "--db", "mem", *extra], input=json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": META}}) + "\n",
        capture_output=True, text=True, env=dict(os.environ, KLEOS_CONFIG="/nonexistent/kleos.json"))
    return json.loads(p.stdout.splitlines()[0])["result"]["tools"]


rows = []
for label, extra in (("MCP tool list (default, 9 tools)", []), ("MCP tool list with --allow-query (10)", ["--allow-query"])):
    ts = tools(extra)
    rows.append((label, tok(json.dumps(ts, separators=(",", ":"))), "every turn"))
    if not extra:
        per = sorted(((t["name"], tok(json.dumps(t, separators=(",", ":")))) for t in ts), key=lambda x: -x[1])

meta_total = body_total = 0
for f in sorted(glob.glob(os.path.join(ROOT, "skills", "*", "SKILL.md"))):
    text = open(f).read()
    front, body = text.split("---", 2)[1], text.split("---", 2)[2]
    meta_total += tok(front)
    body_total += tok(body)
rows.append(("Skills: names + descriptions (4 skills)", meta_total, "every turn"))
rows.append(("Skills: all four bodies (opened on demand)", body_total, "only when used"))

print(f"{'surface':48} {'~tokens':>8}  when in context")
for label, t, when in rows:
    print(f"{label:48} {t:8}  {when}")
print("\nper tool:", ", ".join(f"{n} {t}" for n, t in per))
