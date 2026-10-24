// The same scenario through the Python binding (KLEOS_PY: a python with the
// kleos wheel) and through the TS SDK must give the same normalised results.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { open } from '../dist/index.js';

const PY = process.env.KLEOS_PY;
const skip = !PY && 'set KLEOS_PY to a python with the kleos wheel';

const pyScript = `
import json
from kleos import CozoDbPy
db = CozoDbPy("mem", "", "{}")
db.memory_init()
out = {}
r = db.remember("I moved to Lisbon in March. My sister Anna lives in Munich.", ns="u", entities=["user", "Anna"])
out["action"] = r["action"]; out["claims"] = len(r.get("claims") or [])
out["recall"] = [h["text"] for h in db.recall("Munich", ns="u", k=3)]
out["context_has_anna"] = "Anna" in db.recall_context("Anna", ns="u")["text"]
db.memory_fork("u", "u2")
db.remember("I adopted a cat.", ns="u2")
out["diff_added"] = len(db.memory_diff("u", "u2")["added"])
out["merge_replayed"] = len(db.memory_merge("u2")["replayed"])
out["recall_after_merge"] = sorted(h["text"] for h in db.recall("cat", ns="u", k=5) if "cat" in h["text"])
out["reflect_changes"] = len(db.reflect("u")["changes"])
print(json.dumps(out))
`;

async function tsScenario() {
  const db = open();
  await db.init();
  const out = {};
  const r = await db.remember('I moved to Lisbon in March. My sister Anna lives in Munich.', { ns: 'u', entities: ['user', 'Anna'] });
  out.action = r.action;
  out.claims = (r.claims ?? []).length;
  out.recall = (await db.recall('Munich', { ns: 'u', k: 3 })).map((h) => h.text);
  out.context_has_anna = (await db.recallContext('Anna', { ns: 'u' })).text.includes('Anna');
  await db.fork('u', 'u2');
  await db.remember('I adopted a cat.', { ns: 'u2' });
  out.diff_added = (await db.diff('u', 'u2')).added.length;
  out.merge_replayed = (await db.merge("u2")).replayed.length;
  out.recall_after_merge = (await db.recall('cat', { ns: 'u', k: 5 })).map((h) => h.text).filter((t) => t.includes('cat')).sort();
  out.reflect_changes = (await db.reflect('u')).changes.length;
  await db.close();
  return out;
}

test('parity with the Python binding', { skip }, async () => {
  const py = JSON.parse(execFileSync(PY, ['-I', '-c', pyScript], { encoding: 'utf8' }).trim().split('\n').pop());
  const ts = await tsScenario();
  assert.deepEqual(ts, py);
});
