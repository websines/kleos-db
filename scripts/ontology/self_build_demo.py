"""A company corpus builds its own ontology, with no human input
(docs/specs/ontology.md, "Learning"): notes go through `remember` only;
automatic learning runs every KLEOS_ONTO_LEARN_EVERY untyped claims, then
maintenance runs; the result is printed with its evidence and history.

    KLEOS_ONTO_LEARN_EVERY=30 python self_build_demo.py OUT.json

Real models (EmbeddingGemma 2 or v1, decider-nano) on Metal. The corpus is
generated here from templates (no LLM, no external data).
"""
import json
import os
import random
import sys
import time

from kleos import CozoDbPy

random.seed(20261024)

TEAMS = ["data", "platform", "mobile", "payments", "search"]
VENDORS = ["AWS", "GCP", "Azure", "Hetzner", "Cloudflare"]
CITIES = ["Berlin", "Lisbon", "Austin", "Pune", "Toronto"]
SERVICES = ["checkout", "login", "search API", "billing", "notifications"]


def corpus():
    out = []
    # team headcounts change over time (single-valued, numeric, update cues)
    for t in TEAMS:
        n = random.randint(5, 9)
        for _ in range(4):
            n += random.randint(1, 3)
            out.append(random.choice([
                f"The {t} team now has {n} engineers.",
                f"Headcount update: the {t} team is at {n} engineers now.",
                f"The {t} team grew to {n} engineers this quarter.",
            ]))
    # the cloud vendor of each product line is replaced now and then
    for t in TEAMS[:4]:
        for v in random.sample(VENDORS, 3):
            out.append(random.choice([
                f"The {t} team switched its cloud vendor to {v}.",
                f"Our {t} workloads now run on {v} as the cloud vendor.",
                f"Cloud vendor for {t} is now {v}.",
            ]))
    # incidents accumulate (multi-valued: they never replace each other)
    for s in SERVICES:
        for d in random.sample(["Monday", "Tuesday", "Thursday", "the weekend", "Friday night"], 3):
            out.append(random.choice([
                f"The {s} service had an outage on {d}.",
                f"Incident: {s} was down for an hour on {d}.",
                f"We had a {s} incident on {d}, root cause was a bad deploy.",
            ]))
    # office moves
    for c in CITIES:
        out.append(f"The {c} office moved to a new building downtown.")
        out.append(f"Our {c} office now seats {random.randint(20, 80)} people.")
    # chatter that must stay untouched
    out += ["Thanks everyone for a great offsite!", "Lunch is on the third floor today.",
            "Happy Friday, team!", "Great job on the launch, all."]
    random.shuffle(out)
    return out


def main(out_path):
    db = CozoDbPy("mem", "", "{}")
    emb = os.environ.get("KLEOS_EMBEDDER", "gemma")
    db.enable_embedder(emb, device="metal")
    db.enable_decider("nano", device="metal")
    db.memory_init(embedder=emb, decider="nano")
    notes = corpus()
    t0 = time.time()
    learned_runs = []
    untyped_before = 0
    for i, t in enumerate(notes):
        r = db.remember(t, ns="acme", at=1_767_225_600_000_000 + i * 3_600_000_000)
        untyped_before += sum(1 for x in r.get("types", []) if x["cls"] == "untyped")
        if r.get("ontology_learned"):
            L = r["ontology_learned"]
            learned_runs.append({
                "after_note": i + 1,
                "proposals": [
                    {"id": p["id"], "kind": p["kind"], "status": p["status"], "support": p["support"],
                     "coherence": round(p["coherence"], 3),
                     "prop": (p["body"]["prop"] if isinstance(p["body"]["prop"], str) else
                              {k: p["body"]["prop"][k] for k in ("name", "label", "card", "datatype", "cues")}),
                     "why": p["body"].get("shape", {}).get("why"),
                     "examples": p["body"].get("examples")}
                    for p in L.get("proposals", [])
                ],
                "maintenance": L.get("maintenance"),
            })
    secs = time.time() - t0
    # one more learning pass and maintenance at the end (as a scheduler would)
    final = db.ontology_call("learn", {"ns": "acme"})
    maint = db.ontology_call("maintain", {"ns": "acme"})
    try:
        layer = db.ontology_call("get", {"onto": "learned:acme"})
    except Exception:  # noqa: BLE001
        layer = {"properties": []}
    hist = db.ontology_call("history", {"ns": "acme"})
    typed = db.run_script(
        "?[prop, count(id)] := *kleos_mem_type{ns: 'acme', id, prop}, !is_null(prop)", {}, True)
    untyped_now = db.run_script(
        "?[count(id)] := *kleos_mem_type{ns: 'acme', id, cls}, cls == 'untyped'", {}, True)
    # a new note under a learned single-valued property supersedes the old value
    probe = db.remember("The data team now has 30 engineers.", ns="acme", at=1_767_225_600_000_000 + 999 * 3_600_000_000)
    out = {
        "notes": len(notes),
        "remember_seconds": round(secs, 1),
        "untyped_claims_while_remembering": untyped_before,
        "untyped_now": untyped_now["rows"][0][0],
        "typed_by_property": {r[0]: r[1] for r in typed["rows"]},
        "automatic_runs": learned_runs,
        "final_learn": {"applied": final.get("applied"), "proposals": len(final.get("proposals", []))},
        "maintenance": maint,
        "learned_layer": [{k: p[k] for k in ("name", "label", "card", "datatype", "cues", "doc")} for p in layer["properties"]],
        "history": [{k: h[k] for k in ("kind", "target", "by")} | {"evidence": len(h["evidence"])} for h in hist],
        "probe": {"types": probe.get("types"), "superseded": probe.get("superseded"), "tentative": probe.get("tentative")},
    }
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("notes", "remember_seconds", "untyped_claims_while_remembering", "untyped_now", "typed_by_property")}))
    for p in out["learned_layer"]:
        print("learned:", p["name"], p["card"], p["datatype"], p["cues"])
    print("probe:", json.dumps(out["probe"])[:400])


if __name__ == "__main__":
    main(sys.argv[1])
