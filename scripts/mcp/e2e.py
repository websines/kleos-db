#!/usr/bin/env python3
"""End-to-end check (docs/specs/mcp-cli-skills.md): an MCP client over stdio
(protocol 2026-07-28, stateless) and the `kleos` CLI working on one store.

    python3 scripts/mcp/e2e.py <path-to-kleos-binary> <scratch-dir>
"""
import json, os, subprocess, sys

BIN, SCRATCH = sys.argv[1], sys.argv[2]
os.makedirs(SCRATCH, exist_ok=True)
DB = os.path.join(SCRATCH, "e2e.db")
for f in (DB, DB + "-wal", DB + "-shm"):
    if os.path.exists(f):
        os.remove(f)
ENV = dict(os.environ, KLEOS_CONFIG=os.path.join(SCRATCH, "none.json"))
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientInfo": {"name": "e2e", "version": "1"},
        "io.modelcontextprotocol/clientCapabilities": {"elicitation": {}}}


class Client:
    def __init__(self):
        self.p = subprocess.Popen([BIN, "mcp", "--db", DB, "--ns", "team"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=ENV)
        self.n = 0

    def rpc(self, method, params=None):
        self.n += 1
        params = dict(params or {})
        params["_meta"] = META
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def tool(self, name, args, **extra):
        return self.rpc("tools/call", {"name": name, "arguments": args, **extra})["result"]

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def cli(*args):
    r = subprocess.run([BIN, *args, "--db", DB, "--ns", "team", "--json"], capture_output=True, text=True, env=ENV)
    return json.loads(r.stdout)


steps = []
def check(name, ok, detail=""):
    steps.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))

c = Client()
d = c.rpc("server/discover")["result"]
check("discover advertises 2026-07-28 and tasks", d["supportedVersions"] == ["2026-07-28"] and "io.modelcontextprotocol/tasks" in d["capabilities"]["extensions"])
tools = c.rpc("tools/list")["result"]["tools"]
check("tools/list: 9 tools in a fixed order", [t["name"] for t in tools] == ["recall", "remember", "forget", "reflect", "entity_profile", "doc_add", "doc_status", "flow", "branch"])

r = c.tool("remember", {"text": "Maya leads the payments team."})
mid = r["structuredContent"]["id"]
check("MCP remember", isinstance(mid, str))
r = cli("remember", "The payments team ships on Thursdays.")
check("CLI remember on the same store", r["ok"])
r = c.tool("recall", {"query": "payments team"})
txt = r["content"][0]["text"]
check("MCP recall sees both (CLI write visible to MCP)", "Maya" in txt and "Thursdays" in txt, txt)
r = cli("recall", "payments team")
check("CLI recall sees the MCP write", "Maya" in r["result"]["text"])

# fork in the CLI, write in the fork via MCP, diff via MCP
cli("branch", "fork", "team-try")
c.tool("remember", {"text": "Payments ships on Fridays from now on.", "ns": "team-try"})
r = c.tool("branch", {"action": "diff", "a": "team", "b": "team-try"})
check("branch diff over MCP", len(r["structuredContent"]["added"]) >= 1)

# MRTR confirmation for forget
r = c.tool("forget", {"id": mid})
check("forget asks via MRTR (input_required)", r.get("resultType") == "input_required" and "confirm" in r["inputRequests"])
r = c.tool("forget", {"id": mid}, inputResponses={"confirm": {"action": "accept", "content": {"confirm": True}}}, requestState=r["requestState"])
check("accepted retry forgets", r.get("resultType") == "complete" and not r.get("isError"))
r = cli("recall", "Maya leads", "--list")
check("CLI no longer finds the forgotten memory", all(h["id"] != mid for h in r["result"]))
c.close()

ok = all(s[1] for s in steps)
print(f"\n{sum(s[1] for s in steps)}/{len(steps)} checks passed")
sys.exit(0 if ok else 1)
