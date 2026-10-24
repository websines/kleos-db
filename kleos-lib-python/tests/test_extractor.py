"""Automatic entities from Python (docs/specs/memory-api.md, "Automatic
entities"): enable the built-in GLiNER2.5 extractor, attach it to a memory
store, and let `remember` find the entities. Needs a build with the `infer`
feature and the checkpoint in `KLEOS_GLINER_TEST_MODEL` (skipped otherwise).

Run after `maturin develop --release --features compact,infer-metal`:

    KLEOS_GLINER_TEST_MODEL=<dir> pytest kleos-lib-python/tests/test_extractor.py
"""
import os
import uuid

import pytest

from kleos import CozoDbPy

MODEL = os.environ.get("KLEOS_GLINER_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL or not os.path.isdir(MODEL), reason="set KLEOS_GLINER_TEST_MODEL")


@pytest.fixture
def db():
    name = f"g_{uuid.uuid4().hex[:8]}"
    d = CozoDbPy("mem", "", "{}")
    d.enable_extractor(name, path=MODEL, device=os.environ.get("KLEOS_TEST_DEVICE", "auto"))
    d.memory_init(extractor=name)
    yield d, name
    d.disable_extractor(name)


def names(r):
    return sorted(e["name"] for e in r["entity_links"])


def test_extractors_listed(db):
    d, name = db
    xs = {x["name"]: x for x in d.extractors()}
    assert xs[name]["labels"] == ["person", "location", "organization", "relative"]
    assert xs[name]["version"].startswith("gliner2.5:")


def test_remember_finds_entities_and_relatives(db):
    d, _ = db
    r = d.remember("I live in Paris.", ns="u")
    assert names(r) == ["Paris", "user"]
    assert r["extractor"].startswith("gliner2.5:")
    r = d.remember("My sister Anna moved to Lisbon to work at Google.", ns="u")
    n = names(r)
    assert "Anna" in n and "user's sister" in n and "Lisbon" in n and "Google" in n, n
    assert [(e["src_name"], e["rel"], e["dst_name"]) for e in r["edges"]] == [("user", "sister", "user's sister")]
    # caller entities win; extract: false turns it off
    r = d.remember("I visited Rome.", ns="u", entities=["Rome"])
    assert names(r) == ["Rome"] and "extractor" not in r
    r = d.remember("I visited Oslo.", ns="u", extract=False)
    assert r["entity_links"] == []
