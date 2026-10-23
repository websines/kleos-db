"""The agent memory API from Python (docs/specs/memory-api.md): remember /
recall / forget / reflect / link, with a Python embedder and no decider
(the model-free path: exact duplicates and value rules).

Run after `maturin develop --release`:

    pytest kleos-lib-python/tests/test_memory_api.py
"""
import hashlib
import math
import uuid

import pytest

import kleos
from kleos import CozoDbPy

DAY = 86_400_000_000
T0 = 1_767_225_600_000_000  # 2026-01-01T00:00:00Z in µs


def d(n):
    return T0 + n * DAY


def words(texts):
    """Bag of words hashed into 32 dims, L2-normalised."""
    out = []
    for t in texts:
        v = [0.0] * 32
        for w in "".join(c if c.isalnum() else " " for c in t.lower()).split():
            if len(w) > 2:
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


@pytest.fixture
def db():
    name = "pymem_" + uuid.uuid4().hex[:8]
    kleos.register_embedder(name, words, 32)
    db = CozoDbPy("mem", "", "{}")
    db.memory_init(embedder=name)
    yield db
    kleos.unregister_embedder(name)


def test_remember_recall_and_value_rule_supersede(db):
    a = db.remember("My rent is $1,800 per month.", ns="u", at=d(1), source="chat-1")
    assert a["action"] == "added" and a["embedded"] is True
    db.remember("I like my coffee black.", ns="u", at=d(2))
    b = db.remember("My rent is $2,100 per month.", ns="u", at=d(5))
    assert b["superseded"] == [a["id"]]
    assert b["decisions"][0]["decided_by"].startswith("rules|supersedes@v1|rule")

    now = db.recall("how much is my rent", ns="u", k=5)
    texts = [h["text"] for h in now]
    assert "My rent is $2,100 per month." in texts
    assert "My rent is $1,800 per month." not in texts
    past = db.recall("how much is my rent", ns="u", as_of=d(3))
    old = next(h for h in past if h["id"] == a["id"])
    assert (old["valid_from"], old["valid_to"]) == (d(1), d(5))

    # other namespaces see nothing
    assert db.recall("rent", ns="someone-else") == []


def test_duplicate_forget_reflect_link(db):
    a = db.remember("Alice works at Acme.", ns="u", entities=["Alice", "Acme"])
    dup = db.remember("alice works at ACME", ns="u", source="chat-2")
    assert dup["action"] == "duplicate" and dup["duplicate_of"] == a["id"]
    b = db.remember("Alice's salary is $90,000.", ns="u", entities=["alice"])
    db.remember("Alice's salary is $95,000 now, a raise.", ns="u", entities=["Alice"])
    l = db.link(a["id"], b["id"], "context_for", ns="u")
    assert l["rel"] == "context_for"

    r = db.reflect("u", since=0)
    assert any(c["change"] == "added" for c in r["changes"])
    assert any("90,000" in c["a_text"] for c in r["contradictions"]), r["contradictions"]

    hits = db.recall("works", ns="u", entity="ACME")
    assert [h["text"] for h in hits] == ["Alice works at Acme."]

    f = db.forget(a["id"], ns="u")
    assert f == {"mode": "retract", "ids": [a["id"]]}
    assert all(h["id"] != a["id"] for h in db.recall("Acme", ns="u"))
    e = db.forget(ns="u", entity="alice", mode="erase")
    assert len(e["ids"]) == 2
    rows = db.run_script("?[id] := *kleos_mem{ns: 'u', id}", {}, False)["rows"]
    assert {r[0] for r in rows} == {a["id"]}  # retracted stays, erased are gone


def test_errors_and_cozoscript(db):
    with pytest.raises(Exception, match="nope"):
        db.remember("x", nope=1)
    with pytest.raises(Exception):
        db.forget(ns="u")  # a filter or ids is required
    res = db.run_script("::memory remember 'I live in Berlin.' {ns: 'u'}", {}, False)
    assert res["ok"], res
    res = db.run_script("::memory recall 'Berlin' {ns: 'u'}", {}, True)
    assert res["rows"][0][1] == "I live in Berlin."
