"""Ontologies through the Python binding (docs/specs/ontology.md)."""
from kleos import CozoDbPy


def test_ontology_ops():
    db = CozoDbPy("mem", "", "{}")
    assert any(o["onto"] == "personal" for o in db.ontology_call("list")["ontologies"])
    t = db.ontology_call("type", {"text": "Melanie: Wow, fantastic, Caroline!"})
    assert t["cls"] == "smalltalk"
    db.ontology_call("activate", {"ns": "k", "ontos": ["personal"]})
    db.ontology_call("assert", {"ns": "k", "facts": [
        {"s": "anna", "p": "sister_of", "o": "mom"},
        {"s": "mom", "p": "parent_of", "o": "user"},
    ]})
    inferred = {(f["s"], f["p"], f["o"]) for f in db.ontology_call("infer", {"ns": "k"})}
    assert ("anna", "aunt_of", "user") in inferred
    assert ("user", "child_of", "mom") in inferred


def test_remember_returns_types():
    db = CozoDbPy("mem", "", "{}")
    db.memory_init(ontology="personal", ns="u")
    r = db.remember("I moved to Lisbon last month.", ns="u")
    assert r["types"][0]["prop"] == "lives_in"
    assert r["types"][0]["subject"] == "user"
