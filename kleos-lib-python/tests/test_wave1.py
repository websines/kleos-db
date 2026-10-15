"""Exercise the wave-1 Python surface for real (hybrid fusion/rerank, live
subscriptions, supersession/provenance, half-width vectors + quantized HNSW,
embed()).

Run after `maturin develop --release`:

    pytest kleos-lib-python/tests/test_wave1.py

On-disk databases go under KLEOS_TEST_DIR (default: pytest's tmp_path).
"""
import os
import struct
import threading
import time
import uuid
from pathlib import Path

import pytest

import kleos
from kleos import CozoDbPy

DAY = 86_400_000_000
T0 = 1_767_225_600_000_000  # 2026-01-01T00:00:00Z in µs


def d(n):
    return T0 + n * DAY


@pytest.fixture
def data_dir(tmp_path):
    base = os.environ.get("KLEOS_TEST_DIR")
    if not base:
        return tmp_path
    p = Path(base) / ("t-" + uuid.uuid4().hex[:12])
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(params=["mem", "sqlite"])
def db(request, data_dir):
    if request.param == "mem":
        h = CozoDbPy("mem", "", "{}")
    else:
        h = CozoDbPy("sqlite", str(data_dir / "w1.db"), "{}")
    yield h
    h.close()


def run(db, script, params=None):
    return db.run_script(script, params or {}, False)


def rows(db, script, params=None):
    return run(db, script, params)["rows"]


def wait_for(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def f16(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]


def bf16(x):
    b = struct.pack("<f", x)
    # round-to-nearest-even on the dropped 16 bits
    i = struct.unpack("<I", b)[0]
    i = (i + 0x7FFF + ((i >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", i))[0]


# --------------------------------------------------------------------------
# 1. basics
# --------------------------------------------------------------------------


def test_module_exports():
    assert kleos.__name__ == "kleos"
    for name in ("CozoDbPy", "CozoDbMulTx", "eval_expressions", "variables"):
        assert hasattr(kleos, name), name
    assert kleos.eval_expressions("1 + $x", {"x": 2}, {}) == 3
    assert kleos.variables("a + b * $x", {"x": 1}) == {"a", "b"}


def test_open_and_query(db):
    assert rows(db, "?[x] := x = 1 + 2") == [[3]]
    run(db, ":create kv {k: String => v: Int}")
    run(db, "?[k, v] <- $data :put kv {k => v}", {"data": [["a", 1], ["b", 2]]})
    assert rows(db, "?[k, v] := *kv{k, v} :order k") == [["a", 1], ["b", 2]]
    with pytest.raises(Exception):
        run(db, "?[x] := *nope{x}")


def test_sqlite_persists(data_dir):
    path = str(data_dir / "persist.db")
    db = CozoDbPy("sqlite", path, "{}")
    run(db, ":create kv {k: String => v: Int}")
    run(db, "?[k, v] <- [['a', 1]] :put kv {k => v}")
    assert db.close()
    db = CozoDbPy("sqlite", path, "{}")
    assert rows(db, "?[k, v] := *kv{k, v}") == [["a", 1]]
    db.close()


# --------------------------------------------------------------------------
# 2. hybrid search: fusion + rerank
# --------------------------------------------------------------------------


def make_docs(db):
    run(
        db,
        """
        ?[id, text, emb] <- [
            ['d1', 'the cat sat on the mat', [1.0, 0.0]],
            ['d2', 'a dog ran in the park',  [0.9, 0.1]],
            ['d3', 'cats and dogs are pets', [0.0, 1.0]],
            ['d4', 'the weather is nice',    [0.1, 0.9]]
        ]
        :create docs { id: String => text: String, emb: <F32; 2> }
        """,
    )
    run(
        db,
        "::hnsw create docs:vec {dim: 2, m: 50, dtype: F32, fields: [emb], "
        "distance: Cosine, ef_construction: 20}",
    )
    run(
        db,
        "::fts create docs:fts {extractor: text, tokenizer: Simple, "
        "filters: [Lowercase, Stemmer('English'), Stopwords('en')]}",
    )


def base_query(**kw):
    q = {
        "relation": "docs",
        "id_col": "id",
        "vector_index": "vec",
        "query_vector": [1.0, 0.0],
        "fts_index": "fts",
        "query_text": "cat",
    }
    q.update(kw)
    return q


def ids(res):
    return [r[0] for r in res["rows"]]


def test_hybrid_fusion(db):
    make_docs(db)
    rrf = db.hybrid_search(base_query())
    assert rrf["headers"] == ["id", "score"]
    assert ids(rrf)[0] == "d1"
    assert db.hybrid_search(base_query(fusion="rrf"))["rows"] == rrf["rows"]

    # all weight on the vector leg: the vector order, top score exactly 1
    vec_only = db.hybrid_search(
        base_query(
            fusion={
                "kind": "weighted",
                "weights": {"semantic": 1.0, "text": 0.0},
                "leg_norms": {"semantic": "neg_cosine_distance", "text": "bm25"},
            }
        )
    )
    assert ids(vec_only)[:2] == ["d1", "d2"]
    assert abs(vec_only["rows"][0][1] - 1.0) < 1e-6

    # all weight on text, query "pets": only d3 matches, so d3 first
    text_only = db.hybrid_search(
        base_query(
            query_text="pets",
            fusion={"weights": {"semantic": 0.0, "text": 1.0}, "norm": "minmax"},
        )
    )
    assert ids(text_only)[0] == "d3"
    # ... and the order really changed relative to the vector-only fusion
    assert ids(text_only) != ids(vec_only)

    det = db.hybrid_search(base_query(fusion="weighted", detailed=True))
    assert det["headers"] == ["id", "score", "list_id", "leg_rank", "leg_score"]

    for bad in (
        {"fusion": "nope"},
        {"fusion": {"kind": "nope"}},
        {"fusion": {"weights": {"semantic": 1.0}}},  # missing a configured leg
        {"fusion": {"weights": {"semantic": 1.0, "text": 0.0}, "norm": ["range", 1]}},
    ):
        with pytest.raises(Exception):
            db.hybrid_search(base_query(**bad))


def test_hybrid_rerank(db):
    make_docs(db)
    calls = []

    def word_count(query, docs):
        calls.append((query, list(docs)))
        words = query.split()
        return [float(sum(doc.split().count(w) for w in words)) for doc in docs]

    db.register_reranker("wc", word_count)
    with pytest.raises(Exception):
        db.register_reranker("wc", word_count)  # duplicate name

    fused = ids(db.hybrid_search(base_query()))
    assert fused[0] == "d1"
    res = db.hybrid_search(base_query(rerank={"model": "wc", "query": "weather", "top_n": 4}))
    assert ids(res)[0] == "d4"
    assert len(res["rows"]) == 4
    assert calls and calls[-1][0] == "weather" and len(calls[-1][1]) == 4

    res = db.hybrid_search(base_query(rerank={"model": "wc", "query": "weather", "top_n": 2}))
    assert len(res["rows"]) == 2

    # batching splits the model call
    n = len(calls)
    db.hybrid_search(
        base_query(rerank={"model": "wc", "query": "weather", "top_n": 4, "batch_size": 1})
    )
    assert len(calls) - n == 4

    # composes with weighted fusion + detailed
    res = db.hybrid_search(
        base_query(fusion="weighted", detailed=True, rerank={"model": "wc", "query": "weather"})
    )
    assert res["rows"][0][0] == "d4" and res["rows"][0][1] == 1.0

    # CozoScript Rerank fixed rule with the same Python model
    r = rows(
        db,
        "c[id, text, s] := *docs{id, text}, s = 0.0 "
        "?[id, score] <~ Rerank(c[id, text, s], model: 'wc', query: 'dog park', top_n: 4) "
        ":order -score",
    )
    assert r[0][0] == "d2"

    # a Python exception inside the reranker surfaces as a query error
    def broken(query, docs):
        raise ValueError("boom")

    db.register_reranker("broken", broken)
    with pytest.raises(Exception) as ei:
        db.hybrid_search(base_query(rerank={"model": "broken"}))
    assert "boom" in str(ei.value)

    # wrong-length result is rejected, not truncated
    db.register_reranker("short", lambda q, docs: [1.0])
    with pytest.raises(Exception):
        db.hybrid_search(base_query(rerank={"model": "short", "top_n": 4}))

    assert db.unregister_reranker("wc")
    assert not db.unregister_reranker("wc")
    with pytest.raises(Exception) as ei:
        db.hybrid_search(base_query(rerank={"model": "wc"}))
    assert "wc" in str(ei.value)


def test_rerank_from_other_threads(db):
    """The reranker re-acquires the GIL from inside a GIL-released query."""
    make_docs(db)
    db.register_reranker("len", lambda q, docs: [-float(len(x)) for x in docs])
    errs = []

    def worker():
        try:
            for _ in range(20):
                res = db.hybrid_search(base_query(rerank={"model": "len"}))
                assert ids(res)[0] == "d4"  # shortest text
        except Exception as e:  # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert not any(t.is_alive() for t in ts), "deadlock"
    assert not errs, errs


# --------------------------------------------------------------------------
# 3. live subscriptions
# --------------------------------------------------------------------------


class Collector:
    def __init__(self):
        self.events = []
        self.threads = set()
        self.lock = threading.Lock()

    def __call__(self, ev):
        with self.lock:
            self.events.append(ev)
            self.threads.add(threading.get_ident())

    def kinds(self):
        with self.lock:
            return [e["kind"] for e in self.events]

    def wait(self, n, timeout=10.0):
        assert wait_for(lambda: len(self.events) >= n, timeout), self.events
        with self.lock:
            return list(self.events)


def test_subscribe_initial_and_delta(db):
    run(db, ":create kv {k: String => v: Int}")
    run(db, "?[k, v] <- [['a', 1], ['b', 5]] :put kv {k => v}")
    c = Collector()
    sid = db.subscribe("?[k, v] := *kv{k, v}, v >= $min", {"min": 2}, c)
    assert isinstance(sid, int)
    ev = c.wait(1)[0]
    assert ev["kind"] == "initial"
    assert ev["headers"] == ["k", "v"]
    assert ev["rows"] == [["b", 5]]

    run(db, "?[k, v] <- [['c', 7], ['d', 0]] :put kv {k => v}")
    ev = c.wait(2)[1]
    assert ev["kind"] == "delta"
    assert ev["added"] == [["c", 7]] and ev["removed"] == []

    run(db, "?[k, v] <- [['b', 1]] :put kv {k => v}")
    ev = c.wait(3)[2]
    assert ev["kind"] == "delta"
    assert ev["added"] == [] and ev["removed"] == [["b", 5]]

    # a write the query does not see delivers nothing
    run(db, "?[k, v] <- [['e', -1]] :put kv {k => v}")
    time.sleep(0.2)
    assert len(c.events) == 3
    # the callback ran off the main thread
    assert threading.get_ident() not in c.threads

    assert db.unsubscribe(sid)
    assert not db.unsubscribe(sid)
    run(db, "?[k, v] <- [['f', 9]] :put kv {k => v}")
    time.sleep(0.2)
    assert len(c.events) == 3


def test_subscribe_rejects_bad_scripts(db):
    run(db, ":create kv {k: String => v: Int}")
    for bad in ("?[k] := *kv{k} :put kv {k}", "::relations", "?[x] := *nope{x}", "?[x"):
        with pytest.raises(Exception):
            db.subscribe(bad, {}, lambda ev: None)


def test_subscribe_callback_can_query_and_write(db):
    """The callback queries (and writes to) the db: no GIL deadlock."""
    run(db, ":create kv {k: String => v: Int}")
    run(db, ":create log {n: Int => k: String}")
    seen = []
    done = threading.Event()

    def cb(ev):
        # read inside the callback
        n = rows(db, "?[count(k)] := *kv{k}")[0][0]
        seen.append((ev["kind"], n))
        if ev["kind"] == "delta":
            # write inside the callback (to a relation the query does not read)
            run(db, "?[n, k] <- [[$n, 'x']] :put log {n => k}", {"n": len(seen)})
            if n >= 3:
                done.set()

    sid = db.subscribe("?[k, v] := *kv{k, v}", {}, cb)
    for i in range(3):
        run(db, "?[k, v] <- [[$k, $v]] :put kv {k => v}", {"k": f"k{i}", "v": i})
    assert done.wait(15), seen
    assert seen[0][0] == "initial"
    assert rows(db, "?[count(n)] := *log{n}")[0][0] >= 1
    assert db.unsubscribe(sid)


def test_subscribe_callback_exception_does_not_kill_worker(db, capfd):
    run(db, ":create kv {k: String => v: Int}")
    good = Collector()
    calls = []

    def bad(ev):
        calls.append(ev)
        raise RuntimeError("callback boom")

    def seen_keys(evs):
        out = set()
        for e in evs:
            for r in e.get("rows", []) + e.get("added", []):
                out.add(r[0])
        return out

    s_bad = db.subscribe("?[k] := *kv{k}", {}, bad)
    s_good = db.subscribe("?[k, v] := *kv{k, v}", {}, good)
    for k in ("a", "b", "c"):
        # one write at a time, so deltas are not coalesced: every one must
        # reach the raising callback, i.e. its thread survived the raise
        run(db, "?[k, v] <- [[$k, 1]] :put kv {k => v}", {"k": k})
        assert wait_for(lambda: k in seen_keys(calls)), calls
        assert wait_for(lambda: k in seen_keys(good.events)), good.events
    assert [e["kind"] for e in calls] == ["initial", "delta", "delta", "delta"]
    err = capfd.readouterr().err
    assert "callback boom" in err
    assert db.unsubscribe(s_bad) and db.unsubscribe(s_good)


def test_subscribe_many_threads_and_writers(db):
    run(db, ":create kv {k: Int => v: Int}")
    cols = [Collector() for _ in range(5)]
    sids = [db.subscribe("?[count(k)] := *kv{k}", {}, c) for c in cols]
    # also a distinct query
    other = Collector()
    sids.append(db.subscribe("?[k] := *kv{k}, k < 10", {}, other))

    def writer(base):
        for i in range(20):
            run(db, "?[k, v] <- [[$k, 1]] :put kv {k => v}", {"k": base + i})

    ts = [threading.Thread(target=writer, args=(b * 100,)) for b in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    for c in cols:
        assert wait_for(
            lambda c=c: c.events
            and (c.events[-1].get("rows") or c.events[-1].get("added")) == [[60]]
        ), c.events[-3:]
    assert wait_for(lambda: sum(len(e.get("added", e.get("rows", []))) for e in other.events) >= 10)
    for s in sids:
        assert db.unsubscribe(s)


def test_subscribe_small_queue_resync(db):
    """queue=1 with a slow callback: the subscriber gets a resync, not a stall."""
    run(db, ":create kv {k: Int => v: Int}")
    c = Collector()
    gate = threading.Event()

    def slow(ev):
        gate.wait(10)
        c(ev)

    sid = db.subscribe("?[count(k)] := *kv{k}", {}, slow, queue=1)
    for i in range(30):
        run(db, "?[k, v] <- [[$k, 1]] :put kv {k => v}", {"k": i})
    gate.set()
    assert wait_for(
        lambda: c.events
        and (c.events[-1].get("rows") or c.events[-1].get("added")) == [[30]]
    ), c.events
    assert "resync" in c.kinds()
    assert db.unsubscribe(sid)


def test_unsubscribe_and_close(data_dir):
    db = CozoDbPy("sqlite", str(data_dir / "close.db"), "{}")
    run(db, ":create kv {k: Int => v: Int}")
    c = Collector()
    sid = db.subscribe("?[k] := *kv{k}", {}, c)
    c.wait(1)
    assert db.close()
    assert not db.unsubscribe(sid)  # closed handle: False, no exception
    with pytest.raises(Exception):
        db.subscribe("?[k] := *kv{k}", {}, c)


def test_close_ends_subscriptions(data_dir):
    """close() unsubscribes: the delivery thread exits and drops the callback."""
    import gc
    import weakref

    db = CozoDbPy("sqlite", str(data_dir / "close2.db"), "{}")
    run(db, ":create kv {k: Int => v: Int}")

    class Cb:
        def __init__(self):
            self.n = 0

        def __call__(self, ev):
            self.n += 1

    cb = Cb()
    ref = weakref.ref(cb)
    db.subscribe("?[k] := *kv{k}", {}, cb)
    assert wait_for(lambda: cb.n >= 1)
    del cb
    assert db.close()

    def released():
        gc.collect()
        # pyo3 releases a Py handle dropped off-GIL at the next GIL acquisition
        rows(CozoDbPy("mem", "", "{}"), "?[x] := x = 1")
        return ref() is None

    assert wait_for(released, 5), "subscription survived close()"


def test_many_register_callbacks_do_not_starve_rayon(db):
    """register_callback used to park one rayon worker per callback forever."""
    run(db, ":create e {a: Int, b: Int}")
    got = []
    ids_ = [
        db.register_callback("e", lambda op, new, old: got.append((op, new)))
        for _ in range((os.cpu_count() or 8) * 2 + 2)
    ]
    run(db, "?[a, b] <- [[1, 2], [2, 3], [3, 1]] :put e {a, b}")
    res = {}

    def q():
        res["r"] = rows(db, "?[n, r] <~ PageRank(*e[])")

    t = threading.Thread(target=q, daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "rayon-using query hung"
    assert len(res["r"]) == 3
    assert wait_for(lambda: len(got) >= len(ids_))
    assert got[0][0] == "Put"
    for i in ids_:
        assert db.unregister_callback(i)


# --------------------------------------------------------------------------
# 4. supersession and provenance via scripts
# --------------------------------------------------------------------------


def mem_schema(db):
    run(db, ":create episode {id: String, vt: Validity => text: String}")
    run(
        db,
        ":create fact {subject: String, predicate: String, object: String, "
        "vt: Validity, tt: TxTime => confidence: Float default 1.0}",
    )
    run(db, ":create fact_src {subject: String, predicate: String, object: String, episode: String}")
    run(db, "::supersede create fact {subject, predicate}")
    run(db, "::provenance create fact_src {derived: fact, source: episode}")


def episode(db, eid, day, text):
    run(
        db,
        "?[id, vt, text] <- [[$id, [$t, true], $text]] :put episode {id, vt => text}",
        {"id": eid, "t": d(day), "text": text},
    )


def fact(db, s, p, o, day, eps):
    run(
        db,
        "{?[subject, predicate, object, vt] <- [[$s, $p, $o, [$t, true]]] "
        ":put fact {subject, predicate, object, vt}} "
        "{?[subject, predicate, object, episode] <- $links "
        ":put fact_src {subject, predicate, object, episode}}",
        {"s": s, "p": p, "o": o, "t": d(day), "links": [[s, p, o, e] for e in eps]},
    )


def objects_at(db, s, p, at):
    return sorted(
        r[0]
        for r in rows(
            db,
            "?[o] := *fact{subject: $s, predicate: $p, object: o @ $at}",
            {"s": s, "p": p, "at": at},
        )
    )


def now_us():
    return int(time.time() * 1_000_000)


def test_supersession_and_changes(db):
    mem_schema(db)
    sup = rows(db, "::supersede list")
    assert sup and sup[0][0] == "fact"
    episode(db, "ep1", 1, "Alice joined Acme; lives in Paris")
    fact(db, "alice", "works_at", "Acme", 1, ["ep1"])
    fact(db, "alice", "lives_in", "Paris", 1, ["ep1"])
    episode(db, "ep2", 5, "Alice moved to Berlin")
    fact(db, "alice", "lives_in", "Berlin", 5, ["ep2"])
    episode(db, "ep3", 10, "Alice now works at Globex")
    fact(db, "alice", "works_at", "Globex", 10, ["ep3"])

    assert objects_at(db, "alice", "lives_in", d(3)) == ["Paris"]
    assert objects_at(db, "alice", "lives_in", d(6)) == ["Berlin"]
    assert objects_at(db, "alice", "works_at", d(8)) == ["Acme"]
    assert objects_at(db, "alice", "works_at", d(11)) == ["Globex"]
    # the :put ended Paris without deleting it
    hist = rows(db, "::history fact [['alice', 'lives_in', 'Paris']]")
    assert len(hist) == 2

    ch = run(db, "::changes fact {from: $f, to: $t, prefix: ['alice']}", {"f": d(4), "t": d(11)})
    assert ch["headers"] == ["subject", "predicate", "object", "change", "at", "confidence"]
    got = [(r[2], r[3], r[4]) for r in ch["rows"]]
    assert got == [
        ("Berlin", "added", d(5)),
        ("Paris", "ended", d(5)),
        ("Acme", "ended", d(10)),
        ("Globex", "added", d(10)),
    ]

    # idempotent re-assertion writes nothing
    episode(db, "ep5", 15, "Alice still at Globex")
    fact(db, "alice", "works_at", "Globex", 15, ["ep5"])
    assert len(rows(db, "::history fact [['alice', 'works_at', 'Globex']]")) == 1


def test_provenance_retract_lineage_erase(db):
    mem_schema(db)
    episode(db, "ep2", 5, "Alice moved to Berlin")
    fact(db, "alice", "lives_in", "Berlin", 5, ["ep2"])
    episode(db, "ep3", 10, "Globex")
    episode(db, "ep5", 15, "still Globex")
    fact(db, "alice", "works_at", "Globex", 10, ["ep3", "ep5"])
    assert [r[0] for r in rows(db, "::provenance list")] == ["fact_src"]

    run(db, ":create summary {sid: String, vt: Validity => text: String}")
    run(db, ":create summary_src {sid: String, subject: String, predicate: String, object: String}")
    run(db, "::provenance create summary_src {derived: summary, source: fact}")
    run(
        db,
        "{?[sid, vt, text] <- [['s1', [$t, true], 'Alice lives in Berlin']] :put summary {sid, vt => text}} "
        "{?[sid, subject, predicate, object] <- [['s1', 'alice', 'lives_in', 'Berlin']] "
        ":put summary_src {sid, subject, predicate, object}}",
        {"t": d(6)},
    )

    r = rows(db, "::retract episode [['ep3']]")
    assert r == [[0, "episode", ["ep3"], "retracted"]]
    assert objects_at(db, "alice", "works_at", now_us()) == ["Globex"]

    r = sorted(rows(db, "::retract episode [['ep2']]"))
    assert r == [
        [0, "episode", ["ep2"], "retracted"],
        [1, "fact", ["alice", "lives_in", "Berlin"], "retracted"],
        [2, "summary", ["s1"], "retracted"],
    ]
    assert objects_at(db, "alice", "lives_in", now_us()) == []
    assert objects_at(db, "alice", "lives_in", d(6)) == ["Berlin"]

    lin = run(db, "::lineage summary [['s1']]")
    assert lin["headers"] == ["depth", "relation", "key", "via", "current"]
    assert sorted(lin["rows"]) == [
        [0, "summary", ["s1"], None, False],
        [1, "fact", ["alice", "lives_in", "Berlin"], "summary_src", False],
        [2, "episode", ["ep2"], "fact_src", False],
    ]
    # retract with an explicit valid time
    episode(db, "ep9", 40, "temp")
    fact(db, "bob", "likes", "tea", 40, ["ep9"])
    r = sorted(rows(db, "::retract episode [['ep9']] @ $t", {"t": d(41)}))
    assert [x[3] for x in r] == ["retracted", "retracted"]
    assert objects_at(db, "bob", "likes", d(40) + 1) == ["tea"]
    assert objects_at(db, "bob", "likes", d(42)) == []

    # GDPR erase, with HNSW + FTS on the derived relation
    run(db, ":create note {nid: String, vt: Validity => text: String, emb: <F32; 2>}")
    run(
        db,
        "::hnsw create note:vec {dim: 2, m: 8, dtype: F32, fields: [emb], distance: L2, ef_construction: 32}",
    )
    run(db, "::fts create note:text {extractor: text, tokenizer: Simple, filters: [Lowercase]}")
    run(db, ":create note_src {nid: String, ep: String}")
    run(db, "::provenance create note_src {derived: note, source: episode}")
    episode(db, "ep7", 30, "phone 555-0100")
    episode(db, "ep8", 31, "likes jazz")
    run(
        db,
        "{?[nid, vt, text, emb] <- [['n1', [$a, true], 'phone 555-0100', vec([1.0, 0.0])], "
        "['n2', [$b, true], 'phone and jazz', vec([0.9, 0.1])], "
        "['n3', [$b, true], 'jazz', vec([0.0, 1.0])]] :put note {nid, vt => text, emb}} "
        "{?[nid, ep] <- [['n1', 'ep7'], ['n2', 'ep7'], ['n2', 'ep8'], ['n3', 'ep8']] :put note_src {nid, ep}}",
        {"a": d(30), "b": d(31)},
    )
    knn = "?[nid] := ~note:vec{nid | query: vec([1.0, 0.0]), k: 10, ef: 32}"
    fts = "?[nid] := ~note:text{nid | query: 'phone', k: 10}"
    assert sorted(r[0] for r in rows(db, knn)) == ["n1", "n2", "n3"]
    assert sorted(r[0] for r in rows(db, fts)) == ["n1", "n2"]
    r = sorted(rows(db, "::erase episode [['ep7']]"))
    assert r == [
        [0, "episode", ["ep7"], "erased"],
        [1, "note", ["n1"], "erased"],
        [1, "note", ["n2"], "unlinked"],
    ]
    assert rows(db, "?[v] := *episode{id: 'ep7', vt: v}") == []
    assert rows(db, "?[v] := *note{nid: 'n1', vt: v}") == []
    assert sorted(r[0] for r in rows(db, knn)) == ["n2", "n3"]
    assert sorted(r[0] for r in rows(db, fts)) == ["n2"]
    audit = rows(db, "?[count(r)] := *kleos_evict_audit{relation: r}")
    assert audit[0][0] >= 2


# --------------------------------------------------------------------------
# 5. vectors: F16 / BF16, quantized + filtered HNSW
# --------------------------------------------------------------------------


def test_half_width_round_trip(db):
    run(db, ":create h { id: Int => a: <F16; 3>, b: <BF16; 3>?, c: [<F16; 2>]? }")
    cols = rows(db, "::columns h")
    types = [str(r) for r in cols]
    assert any("<F16;3>" in t for t in types), types
    assert any("<BF16;3>?" in t for t in types), types
    run(
        db,
        "?[id, a, b, c] <- $rows :put h {id => a, b, c}",
        {"rows": [[1, [1.0, 0.1, 65504.0], [1.0, 0.1, 3.0], [[0.5, 0.3]]], [2, [2.0, -0.333, 1e-3], None, None]]},
    )
    r = rows(db, "?[id, a, b, c] := *h{id, a, b, c} :order id")
    assert r[0][1] == [1.0, f16(0.1), 65504.0]
    assert r[0][2] == [1.0, bf16(0.1), 3.0]
    assert r[0][3] == [[0.5, f16(0.3)]]
    assert r[1][1] == [2.0, f16(-0.333), f16(1e-3)]
    assert r[1][2] is None
    # values read back into Python can be written back verbatim
    run(db, "?[id, a, b, c] <- [[4, $a, $b, $c]] :put h {id => a, b, c}", {"a": r[0][1], "b": r[0][2], "c": r[0][3]})
    assert rows(db, "?[a, b, c] := *h{id: 4, a, b, c}")[0] == r[0][1:]
    # vec(..., 'F16') literal
    run(db, "?[id, a, b, c] <- [[3, vec([1.5, 2.5, 3.5], 'F16'), null, null]] :put h {id => a, b, c}")
    assert rows(db, "?[a] := *h{id: 3, a}") == [[[1.5, 2.5, 3.5]]]
    # wrong width is rejected
    with pytest.raises(Exception):
        run(db, "?[id, a, b, c] <- [[5, [1.0, 2.0], null, null]] :put h {id => a, b, c}")


def lcg_vectors(n, dim, seed):
    x = seed
    out = []
    for _ in range(n):
        v = []
        for _ in range(dim):
            x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
            v.append(((x >> 33) / (1 << 31)) * 2.0 - 1.0)
        out.append(v)
    return out


def l2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


@pytest.mark.parametrize("dtype", ["F32", "F16"])
def test_hnsw_quantized_and_filtered(db, dtype):
    dim, n = 16, 400
    pts = lcg_vectors(n, dim, 7)
    run(db, f":create pts {{ id: Int => cat: Int, emb: <{dtype}; {dim}> }}")
    run(
        db,
        f"::hnsw create pts:idx {{ dim: {dim}, m: 12, dtype: {dtype}, fields: [emb], "
        f"distance: L2, ef_construction: 64, quantize: 'turbo4' }}",
    )
    run(
        db,
        "?[id, cat, emb] <- $rows :put pts {id => cat, emb}",
        {"rows": [[i, i % 10, p] for i, p in enumerate(pts)]},
    )
    # the index stores code rows
    codes = rows(db, "?[count(fr_id)] := *pts:idx{layer: 2, fr_id}")
    assert codes and codes[0][0] > 0

    stored = {r[0]: r[1] for r in rows(db, "?[id, emb] := *pts{id, emb}")}
    queries = lcg_vectors(10, dim, 99)
    recall = 0.0
    for q in queries:
        res = rows(
            db,
            "?[id, d] := ~pts:idx{id | query: vec($q), k: 10, ef: 64, bind_distance: d}",
            {"q": q},
        )
        assert len(res) == 10
        truth = sorted(range(n), key=lambda i: l2(q, stored[i]))[:10]
        recall += len(set(r[0] for r in res) & set(truth)) / 10
        # returned distances are exact (computed on the stored vectors);
        # loose tolerance: F16 queries/stored values are rounded
        for i, dist in res:
            assert abs(dist - l2(q, stored[i])) <= 1e-2 * max(1.0, l2(q, stored[i])), (i, dist)
    assert recall / len(queries) >= 0.8, recall / len(queries)

    # filter: only cat == 3 rows, still k results
    for q in queries[:5]:
        res = rows(
            db,
            "?[id, cat] := ~pts:idx{id, cat | query: vec($q), k: 10, ef: 32, filter: cat == 3}",
            {"q": q},
        )
        assert len(res) == 10
        assert all(c == 3 for _, c in res)
        truth = sorted((i for i in range(n) if i % 10 == 3), key=lambda i: l2(q, stored[i]))[:10]
        assert len(set(r[0] for r in res) & set(truth)) >= 8

    # a filter nothing passes
    assert rows(db, "?[id] := ~pts:idx{id, cat | query: vec($q), k: 5, ef: 32, filter: cat > 100}", {"q": queries[0]}) == []

    # bad quantize spec
    with pytest.raises(Exception):
        run(
            db,
            f"::hnsw create pts:bad {{ dim: {dim}, m: 8, dtype: {dtype}, fields: [emb], "
            "distance: L2, ef_construction: 16, quantize: turbo5 }",
        )


# --------------------------------------------------------------------------
# 6. embed() with a Python-registered embedder
# --------------------------------------------------------------------------


def toy_embed(texts):
    # 4-d bag of letters, deterministic
    out = []
    for t in texts:
        v = [0.0, 0.0, 0.0, 0.0]
        for ch in t:
            v[ord(ch) % 4] += 1.0
        out.append(v)
    return out


needs_embedder = pytest.mark.skipif(
    not hasattr(kleos, "register_embedder"), reason="bindings expose no register_embedder"
)


@needs_embedder
def test_embed_function_and_computed_column(db):
    calls = []

    def emb(texts):
        calls.append(list(texts))
        return toy_embed(texts)

    name = "py_toy_" + uuid.uuid4().hex[:6]
    kleos.register_embedder(name, emb, 4)
    try:
        assert (name, 4) in [tuple(x) for x in kleos.registered_embedders()]
        v = rows(db, f"?[v] := v = embed('{name}', 'abcd')")[0][0]
        assert len(v) == 4
        # cached: a second call does not reach Python
        n = len(calls)
        rows(db, f"?[v] := v = embed('{name}', 'abcd')")
        assert len(calls) == n

        run(db, f":create docs {{ id: Int => text: String, vec: <F32; 4> default embed('{name}', text) }}")
        run(
            db,
            "?[id, text] <- $rows :put docs {id => text}",
            {"rows": [[i, f"doc number {i} " + "x" * i] for i in range(20)]},
        )
        r = rows(db, "?[id, vec] := *docs{id, vec} :order id")
        assert len(r) == 20 and all(len(x[1]) == 4 for x in r)
        assert r[3][1] == [float(x) for x in toy_embed(["doc number 3 xxx"])[0]]
        run(db, "?[id, text] <- [[3, 'zzzz']] :update docs {id => text}")
        assert rows(db, "?[vec] := *docs{id: 3, vec}")[0][0] == toy_embed(["zzzz"])[0]

        # errors from Python and wrong widths surface as query errors
        bad = name + "_bad"
        kleos.register_embedder(bad, lambda texts: [[1.0]] * len(texts), 4)
        with pytest.raises(Exception):
            rows(db, f"?[v] := v = embed('{bad}', 'q')")
        kleos.register_embedder(bad, lambda texts: 1 / 0, 4)
        with pytest.raises(Exception) as ei:
            rows(db, f"?[v] := v = embed('{bad}', 'q2')")
        assert "division" in str(ei.value)
        assert kleos.unregister_embedder(bad)
    finally:
        assert kleos.unregister_embedder(name)
        assert not kleos.unregister_embedder(name)
    with pytest.raises(Exception):
        rows(db, f"?[v] := v = embed('{name}', 'new text')")


@needs_embedder
def test_embed_hnsw_and_threads(db):
    name = "py_hnsw_" + uuid.uuid4().hex[:6]
    kleos.register_embedder(name, toy_embed, 4, max_batch=3)
    try:
        run(
            db,
            f":create mem {{ id: Int => text: String, vec: <F32; 4> default embed('{name}', text) }}",
        )
        run(
            db,
            "::hnsw create mem:v { dim: 4, m: 8, dtype: F32, fields: [vec], distance: L2, ef_construction: 16 }",
        )
        texts = ["aaaa", "bbbb", "cccc", "dddd", "abab", "cdcd"]
        run(db, "?[id, text] <- $rows :put mem {id => text}", {"rows": list(enumerate(texts))})
        r = rows(
            db,
            f"?[id] := ~mem:v{{ id | query: embed('{name}', 'bbbb'), k: 1, ef: 16 }}",
        )
        assert r == [[1]]
        # embed() from several Python threads at once (GIL re-acquired inside
        # GIL-released queries)
        errs = []

        def worker(j):
            try:
                for i in range(10):
                    v = rows(db, f"?[v] := v = embed('{name}', $t)", {"t": f"t{j}-{i}"})[0][0]
                    assert v == toy_embed([f"t{j}-{i}"])[0]
            except Exception as e:  # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=worker, args=(j,)) for j in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
        assert not any(t.is_alive() for t in ts), "deadlock"
        assert not errs, errs
        # module-level eval_expressions keeps the GIL; embed() must still work
        assert kleos.eval_expressions(f"embed('{name}', 'ab')", {}, {}) == toy_embed(["ab"])[0]
    finally:
        kleos.unregister_embedder(name)
