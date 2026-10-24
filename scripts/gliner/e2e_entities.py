# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Phase 2: the fine-tuned GLiNER2.5 base as `remember(entities=...)` for the
end-to-end runs (docs/research/gliner.md, "Phase 2").

    python e2e_entities.py locomo <out.json> [ckpt] [mps|cpu]     # per LoCoMo turn, by sample_id/dia_id
    python e2e_entities.py ku <out.json> [ckpt] [mps|cpu]         # per user turn of the 23 held-out KU questions
    python e2e_entities.py life <life_stream.json> <out.json> ... # the life stream, entities replaced

What a statement's entity list holds:
- **Names:** person, location and organization spans above the per-label
  thresholds tuned on the silver validation split (ft_report.json,
  ft-base-ep3), after the training-label cleanup (a capital letter, not a
  pronoun).
- **The speaker** when the text is first person (the regex that beat the
  model on "about the speaker": I, me, my, we, our, ...), not counting the
  possessive inside a relation span ("my sister Anna lives in Munich" is
  about Anna). For a single-user
  stream (LongMemEval, the life stream) the speaker is `user`; in LoCoMo two
  people share a namespace, so it is the speaker's name ("Caroline"), which
  is also how kleos's claims carry the "Name:" prefix.
- **Relations** as stable keys anchored on whose relation it is: "my sister"
  said by Caroline -> `Caroline.sister`; "Melanie's kids" -> `Melanie.kids`;
  a bare kin word ("Mom said") is the speaker's. "his/her/their X" is skipped
  (whose is unknown from one turn).
- **Relation + name links:** "my sister Anna" gives `Caroline.sister` and
  `Anna`, and the pair is remembered per namespace in time order, so a later
  turn mentioning only one of them carries both (kleos's resolver cannot know
  that `Caroline.sister` is Anna; the same-subject gate needs them to share).
- **Not the listener:** a person name in vocative position ("Hey Deborah!",
  "Thanks, Melanie!") is dropped; LoCoMo turns address the other speaker
  constantly, and that name says nothing about what the turn is about.
  Pairs never link a speaker or a group ("my family").
GLiNER's own time per text is recorded (`ms`, amortised over the batch).
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("GLINER_DIR", "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/gliner-ft")
import common as C  # noqa: E402
from ft_eval import FIRST_PERSON, norm, pairs_rule, post  # noqa: E402
from ft_data import ENTS, KINDS  # noqa: E402
from ft_train import ABOUT  # noqa: E402

CKPT = "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/gliner-ft/ckpt/base-e3/final"
LOW = 0.1
LOCOMO = Path(os.environ.get("LOCOMO_JSON", "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/dates/data/locomo/locomo10.json"))


def thresholds():
    r = json.loads((C.GDIR / "ft_report.json").read_text())
    return r["ft-base-ep3"]["thresholds"]


def load(ckpt, device):
    from gliner2 import AutoExtractor
    m = AutoExtractor.from_pretrained(ckpt, map_location=device)
    m.eval()
    # the fine-tuned model was trained on bare label names with this joint schema
    s = m.create_schema().entities(ENTS).classification("kind", KINDS).classification("about_speaker", ABOUT)
    return m, s


def predict(m, s, texts, bs=8):
    t0 = time.perf_counter()
    res = m.batch_extract(texts, s, batch_size=bs, threshold=LOW, include_confidence=True, include_spans=True)
    ms = 1000 * (time.perf_counter() - t0) / max(1, len(texts))
    out = []
    for r in res:
        ents = [{"label": lab, "text": e["text"], "start": e.get("start"), "end": e.get("end"), "p": e.get("confidence", 1.0)}
                for lab, lst in (r.get("entities") or {}).items() for e in lst]
        out.append(ents)
    return out, ms


_POSS = re.compile(r"^(my|our)\s+(.+)$", re.I)
_OTHER = re.compile(r"^(his|her|their|your)\s+(.+)$", re.I)
_NAMED = re.compile(r"^([A-Z][\w-]+)'s\s+(.+)$")


def rel_key(span, speaker):
    t = span.strip(" ,.!?\"'")
    m = _NAMED.match(t)
    if m:
        return f"{m.group(1)}.{norm(m.group(2)).replace(' ', '_')}"
    if _OTHER.match(t):
        return None
    m = _POSS.match(t)
    rel = norm(m.group(2) if m else t)
    if not rel or len(rel.split()) > 3:
        return None
    return f"{speaker}.{rel.replace(' ', '_')}"


GREET = {"hey", "hi", "hello", "thanks", "thank", "you", "wow", "yeah", "yes", "oh", "congrats", "congratulations",
         "sorry", "ok", "okay", "aww", "aw", "oof", "ugh", "haha", "lol", "dear", "bye", "see", "night", "morning"}
PLURAL_REL = {"family", "fam", "folks", "kids", "children", "parents", "friends", "pets", "siblings", "team", "teammates"}


def vocative(text, e):
    """A name used to address someone ("Hey Deborah!", "Thanks, Melanie!",
    "that's tough, Jon.") names the listener, not what the turn is about."""
    if e.get("start") is None:
        return False
    pre, post_ = text[:e["start"]].rstrip(), text[e["end"]:].lstrip()
    if post_[:1] not in ("!", "?", ",", ".", "") and not post_.startswith(":)"):
        return False
    return pre == "" or pre[-1] in ",.!?" or pre.split()[-1].lower().strip(",.!") in GREET


def entities_for(text, ents, speaker, thr, links, speakers=()):
    """One statement's entity list (see the module doc). `links` is the
    namespace's relation<->name map, updated in place."""
    kept = [e for e in ents if e["p"] >= thr[e["label"]]]
    p = post({"entities": kept})
    out = []

    def add(x):
        if x and x not in out:
            out.append(x)

    # "my sister Anna lives in Munich" is about the sister, not the speaker:
    # the possessive inside a relation span does not count as first person
    bare = list(text)
    for e in kept:
        if e["label"] == "relative" and e.get("start") is not None:
            bare[e["start"]:e["end"]] = " " * (e["end"] - e["start"])
    if FIRST_PERSON.search("".join(bare)):
        add(speaker)
    rels = {}
    for e in p["entities"]:
        if e["label"] == "relative":
            k = rel_key(e["text"], speaker)
            if k:
                rels[norm(e["text"])] = k
                add(k)
        elif norm(e["text"]) != norm(speaker) and not (e["label"] == "person" and vocative(text, e)):
            add(e["text"].strip(" ,.!?\"'"))
    pairs = set(p["_pairs"]) | pairs_rule(text, p["entities"])
    for rel, name in pairs:
        k = rels.get(rel) or rel_key(rel, speaker)
        # link only a single relative to a third person: never a speaker of
        # the conversation, never a group ("my family")
        if not k or k.split(".", 1)[1] in PLURAL_REL or k.endswith("s") or name in {norm(x) for x in (speaker, *speakers)}:
            continue
        nm = next((x for x in out if norm(x) == name), name.title())
        links.setdefault(k, set()).add(nm)
        links.setdefault(norm(nm), set()).add(k)
    for x in list(out):
        for y in sorted(links.get(x, set()) | links.get(norm(x), set())):
            add(y)
    return out, sorted(pairs)


def run_locomo(out_path, ckpt, device):
    m, s = load(ckpt, device)
    thr = thresholds()
    data = json.loads(LOCOMO.read_text())
    res, all_ms = {}, []
    for conv in data:
        c = conv["conversation"]
        turns, n = [], 1
        while f"session_{n}" in c:
            turns += [(t["dia_id"], t["speaker"], t["text"]) for t in c[f"session_{n}"]]
            n += 1
        preds, ms = predict(m, s, [t[2] for t in turns])
        all_ms.append((ms, len(turns)))
        links = {}
        spks = (c["speaker_a"], c["speaker_b"])
        for (dia, spk, text), ents in zip(turns, preds):
            e, pairs = entities_for(text, ents, spk, thr, links, spks)
            res[f"{conv['sample_id']}/{dia}"] = {"entities": e, "pairs": pairs, "raw": [x for x in ents if x["p"] >= thr[x["label"]]], "ms": ms}
        print(conv["sample_id"], len(turns), f"{ms:.1f} ms/turn", flush=True)
    tot = sum(n for _, n in all_ms)
    meta = {"ckpt": ckpt, "device": device, "thresholds": thr, "ms_per_text": sum(a * n for a, n in all_ms) / tot, "n": tot}
    Path(out_path).write_text(json.dumps({"meta": meta, "items": res}))
    print(json.dumps(meta))


def key(text):
    return hashlib.sha1(text.encode()).hexdigest()


def run_ku(out_path, ckpt, device):
    m, s = load(ckpt, device)
    thr = thresholds()
    raw = C.lme()
    data = {x["question_id"]: x for x in raw}
    held, _ = C.lme_ku_split(raw)
    res, tot_ms, n = {}, 0.0, 0
    for qid in sorted(q for q in held if q in data and not q.endswith("_abs")):
        q = data[qid]
        sess = sorted(zip(q["haystack_dates"], q["haystack_sessions"]), key=lambda z: C.micros(z[0]))
        texts = [t["content"] for _, ss in sess for t in ss if t["role"] == "user"]
        preds, ms = predict(m, s, texts)
        tot_ms += ms * len(texts)
        n += len(texts)
        links = {}
        for text, ents in zip(texts, preds):
            e, pairs = entities_for(text, ents, "user", thr, links)
            res[f"{qid}/{key(text)}"] = {"entities": e, "pairs": pairs, "ms": ms}
    meta = {"ckpt": ckpt, "device": device, "thresholds": thr, "ms_per_text": tot_ms / n, "n": n}
    Path(out_path).write_text(json.dumps({"meta": meta, "items": res}))
    print(json.dumps(meta))


def run_life(in_path, out_path, ckpt, device):
    m, s = load(ckpt, device)
    thr = thresholds()
    data = json.loads(Path(in_path).read_text())
    preds, ms = predict(m, s, [t["text"] for t in data["turns"]])
    links = {}
    for t, ents in zip(data["turns"], preds):
        t["entities_hand"] = t["entities"]
        t["entities"], _ = entities_for(t["text"], ents, "user", thr, links)
    data["gliner"] = {"ckpt": ckpt, "device": device, "thresholds": thr, "ms_per_text": ms}
    Path(out_path).write_text(json.dumps(data, indent=1, ensure_ascii=False))
    for t in data["turns"]:
        print(f"{t['text'][:60]:60s} hand={t['entities_hand']} gliner={t['entities']}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "life":
        run_life(sys.argv[2], sys.argv[3], *(sys.argv[4:6] or [CKPT, "mps"]))
    else:
        rest = sys.argv[3:5]
        ckpt = rest[0] if rest else CKPT
        dev = rest[1] if len(rest) > 1 else "mps"
        (run_locomo if mode == "locomo" else run_ku)(sys.argv[2], ckpt, dev)
