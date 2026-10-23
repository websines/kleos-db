"""Entity resolution from Python (docs/specs/entity-resolution.md), without
models: the rule tier plus LSH and FTS, merges, split and history.

Run after `maturin develop --release`:

    pytest kleos-lib-python/tests/test_entity.py
"""
import pytest

from kleos import CozoDbPy


@pytest.fixture
def db():
    d = CozoDbPy("mem", "", "{}")
    d.run_script(
        ":create person {id: String => name: String, type: String? default null, description: String? default null}",
        {},
        False,
    )
    d.run_script(
        "::lsh create person:lsh {extractor: name, tokenizer: NGram(3, 3, false), filters: [Lowercase], n_perm: 64, target_threshold: 0.3, n_gram: 1}",
        {},
        False,
    )
    d.run_script(
        "?[id, name, type, description] <- [['p1', 'Alice Smith', 'person', 'manager at Acme'], ['p2', 'Robert Jones', 'person', null]] :put person {id => name, type, description}",
        {},
        False,
    )
    return d


OPTS = dict(type="type", description="description", lsh="lsh")


def test_resolve_rules_and_commit(db):
    res = db.resolve_entities(
        "person",
        ["Dr. Alice Smith", {"name": "A. Smith", "type": "person", "context": "my manager", "id": "m1"}, "Zed Quux"],
        commit=True,
        **OPTS,
    )
    assert [r["outcome"] for r in res] == ["match", "possible", "new"]
    assert res[0]["entity"] == "p1"
    assert res[0]["decided_by"] == "rule|exact"
    assert res[1]["candidates"][0]["rule"] == "initial"
    assert res[1]["id"] == "m1"
    pending = db.entity_pending("person")
    assert pending["rows"][0][:2] == ["m1", "p1"]


def test_merge_split_history(db):
    db.run_script("?[id, name, type] <- [['m1', 'Al Smith', 'person']] :put person {id => name, type}", {}, False)
    db.merge_entities("person", "m1", "p1")
    assert db.canonical_entity("person", "m1") == "p1"
    assert db.split_entity("person", "m1") == "p1"
    assert db.canonical_entity("person", "m1") == "m1"
    assert db.split_entity("person", "m1") is None
    hist = db.entity_history("person", "m1")
    assert [h["action"] for h in hist] == ["merge", "split"]
    assert hist[0]["decided_by"] == "manual"
    with pytest.raises(Exception):
        db.resolve_entities("person", ["x"], colour="red")
