"""Build the decider-nano fine-tuning data (reproducible, seeded).

    HF_HOME=/Volumes/SSD01/caches/huggingface .bench-data/infer/.venv/bin/python \
        scripts/infer-bench/decide/train/build_data.py [OUT_DIR]

OUT_DIR defaults to .bench-data/nano-tune/data. Writes
  train.jsonl / val.jsonl   one rendered request per line: state, question, options, allowed
                            (indices of the correct option(s)), src, cls, group
  lme_ku_heldout.json       LongMemEval knowledge-update held-out pairs (eval only)
  mnli_dev.json             MNLI validation_matched subset (forgetting check, eval only)
  manifest.json             counts, licences, dedup report

Sources and licences (see NOTICE.md):
  tals/vitaminc train split          CC BY-SA 3.0   (test split: eval only, never read here
                                                     except to dedup against the 300-item sample)
  xiaowu0162/longmemeval-cleaned     MIT            (oracle file; questions split train/held-out)
  google-research-datasets/paws      PAWS licence (free use, attribution to Google)
  nyu-mll/multi_nli                  OANC / CC BY-SA 3.0 / CC BY 3.0 per genre
  programmatic (memgen.py)           kleos authors

Never trained on: the 60 adapt + 42 eval + 24 entity items (kleos-core/examples/data/
decide_items.json), the VitaminC-300 sample (.bench-data/decide/vitaminc_sample.json, test300 and
train100), and the LongMemEval held-out pairs. Training pairs that are exact or near duplicates of any
of them (token Jaccard) are dropped; the manifest reports how many.
"""
import glob
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kleos_templates as KT  # noqa: E402
import memgen  # noqa: E402

ROOT = HERE.parents[3]                      # the kleos checkout
BENCH = ROOT.parent.parent / ".bench-data" if (ROOT.parent.name == ".worktrees") else ROOT.parent / ".bench-data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else BENCH / "nano-tune" / "data"
HUB = Path("/Volumes/SSD01/caches/huggingface/hub")
SEED = 20261023
rng = random.Random(SEED)


def snap(repo):
    return Path(glob.glob(str(HUB / f"datasets--{repo.replace('/', '--')}" / "snapshots" / "*"))[0])


# ------------------------------------------------------------------ held-out texts (never train)
items = json.loads((ROOT / "kleos-core/examples/data/decide_items.json").read_text())
vit_sample = json.loads((BENCH / "decide/vitaminc_sample.json").read_text())
HELD_PAIRS = []
for k in ("adapt", "eval", "entity"):
    HELD_PAIRS += [(a, b) for a, b, _ in items[k]]
for k in ("test300", "train100"):
    HELD_PAIRS += [(a, b) for a, b, _ in vit_sample[k]]

WORD = re.compile(r"[a-z0-9]+")


def toks(t):
    return frozenset(WORD.findall(t.lower()))


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


HELD_TEXTS = {}
for a, b in HELD_PAIRS:
    for t in (a, b):
        HELD_TEXTS[" ".join(WORD.findall(t.lower()))] = toks(t)
HELD_TOKS = list(HELD_TEXTS.values())
HELD_PAIR_TOKS = [(toks(a), toks(b)) for a, b in HELD_PAIRS]
INDEX = defaultdict(list)  # rare-ish token -> held text ids, for fast near-dup search
for i, t in enumerate(HELD_TOKS):
    for w in t:
        INDEX[w].append(i)


def near_held(text, thr=0.6):
    """True when `text` is an exact or near duplicate (token Jaccard ≥ thr) of a held-out text."""
    t = toks(text)
    if " ".join(WORD.findall(text.lower())) in HELD_TEXTS:
        return True
    cand = set()
    for w in t:
        if len(INDEX.get(w, ())) < 40:
            cand.update(INDEX.get(w, ()))
    return any(jac(t, HELD_TOKS[i]) >= thr for i in cand)


def pair_near_held(a, b, thr=0.5):
    ta, tb = toks(a), toks(b)
    for ha, hb in HELD_PAIR_TOKS:
        if (jac(ta, ha) + jac(tb, hb)) / 2 >= thr or (jac(ta, hb) + jac(tb, ha)) / 2 >= thr:
            return True
    return False


# ------------------------------------------------------------------ record builders
def rec(state, question, options, allowed, src, cls, group, a=None, b=None):
    return dict(state=state, question=question, options=options, allowed=allowed, src=src, cls=cls, group=group,
                a=a, b=b)


def nli_recs(a, b, cls, src, group, both_orders=True, with_relation=True):
    """Memory relation class -> NLI requests (supersedes / contradicts / duplicate templates) and the
    CNLI relation request."""
    out = []
    orders = [(a, b), (b, a)] if both_orders else [(a, b)]
    for x, y in orders:
        s, q, o = KT.nli(x, y)
        out.append(rec(s, q, o, KT.NLI_TARGET[cls], src, cls, group, x, y))
    if with_relation:
        s, q, o = KT.cnli(a, b)
        out.append(rec(s, q, o, KT.CNLI_TARGET[cls], src + "/rel", cls, group, a, b))
    return out


records = []
report = {"dropped_near_heldout": Counter()}


def add(recs, check_pairs=True):
    for r in recs:
        if r["a"] is not None and check_pairs and (near_held(r["a"]) and near_held(r["b"]) or pair_near_held(r["a"], r["b"])):
            report["dropped_near_heldout"][r["src"]] += 1
            continue
        records.append(r)


# ------------------------------------------------------------------ 1. VitaminC train (CC BY-SA 3.0)
vdir = snap("tals/vitaminc")
vit = [json.loads(l) for l in open(vdir / "train.jsonl")]
held_claims = {" ".join(WORD.findall(a.lower())) for a, _ in HELD_PAIRS}
held_evid = {" ".join(WORD.findall(b.lower())) for _, b in HELD_PAIRS}
vit = [r for r in vit if " ".join(WORD.findall(r["claim"].lower())) not in held_claims
       and " ".join(WORD.findall(r["evidence"].lower())) not in held_evid]
rng.shuffle(vit)
VMAP = {"SUPPORTS": "duplicate", "REFUTES": "update", "NOT ENOUGH INFO": "compatible"}
VTGT = {"SUPPORTS": [KT.ENT], "REFUTES": [KT.CON], "NOT ENOUGH INFO": [KT.NEU]}
per = Counter()
N_VIT_CE = 900  # per label: claim (old) / evidence (new), the VitaminC-300 framing
for r in vit:
    lab = r["label"]
    if per[lab] >= N_VIT_CE or len(r["evidence"].split()) > 90:
        continue
    per[lab] += 1
    s, q, o = KT.nli(r["claim"], r["evidence"])
    add([rec(s, q, o, VTGT[lab], "vitaminc/claim-evidence", VMAP[lab], "vc" + r["case_id"], r["claim"], r["evidence"])])
# revision pairs: the two evidence sentences of one real revision case, where a claim's label flips
by_case = defaultdict(list)
for r in vit:
    if r["revision_type"] == "real":
        by_case[r["case_id"]].append(r)
n_rev = 0
for cid, rows in by_case.items():
    ev = sorted({r["evidence"] for r in rows})
    if len(ev) != 2 or any(len(e.split()) > 70 for e in ev):
        continue
    labs = defaultdict(set)
    for r in rows:
        labs[r["claim"]].add((r["evidence"], r["label"]))
    flips = any(len({l for _, l in v}) > 1 for v in labs.values())
    if not flips:
        continue
    old, new = ev if rng.random() < 0.5 else ev[::-1]
    add(nli_recs(old, new, "update", "vitaminc/revision", "vr" + cid, both_orders=False, with_relation=rng.random() < 0.3))
    n_rev += 1
    if n_rev >= 700:
        break

# ------------------------------------------------------------------ 2. LongMemEval knowledge-update (MIT)
lme = json.loads(open(snap("xiaowu0162/longmemeval-cleaned") / "longmemeval_oracle.json").read())
SENT = re.compile(r"(?<=[.!?])\s+")


def clip_turn(turn, other, max_words=70):
    """Keep a long chat turn's sentences that share most content words with the paired turn."""
    if len(turn.split()) <= max_words:
        return turn
    sents = SENT.split(turn)
    ot = toks(other)
    scored = sorted(range(len(sents)), key=lambda i: -len(toks(sents[i]) & ot))
    keep = sorted(scored[:2])
    out = " ".join(sents[i] for i in keep)
    return " ".join(out.split()[:max_words])


def answer_turns(x):
    """[(date, text)] user turns with has_answer, ordered by session date."""
    out = []
    for sess, d in zip(x["haystack_sessions"], x["haystack_dates"]):
        for t in sess:
            if t.get("has_answer") and t["role"] == "user":
                out.append((d, t["content"]))
    return sorted(out, key=lambda z: z[0])


def lme_pairs(x, positive):
    ts = answer_turns(x)
    pairs = []
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            if ts[i][0] == ts[j][0]:
                continue
            a, b = ts[i][1], ts[j][1]
            pairs.append((clip_turn(a, b), clip_turn(b, a)))
    return pairs[:3]


ku = sorted([x for x in lme if x["question_type"] == "knowledge-update"], key=lambda x: x["question_id"])
ms = sorted([x for x in lme if x["question_type"] == "multi-session"], key=lambda x: x["question_id"])
lrng = random.Random(SEED + 1)
lrng.shuffle(ku)
lrng.shuffle(ms)
ku_held, ku_train = ku[:25], ku[25:]
ms_held, ms_train = ms[:40], ms[40:]
heldout_lme = []
for x in ku_held:
    heldout_lme += [(a, b, "update", x["question_id"]) for a, b in lme_pairs(x, True)]
for x in ms_held:
    heldout_lme += [(a, b, "compatible", x["question_id"]) for a, b in lme_pairs(x, False)]
# the held-out pairs join the never-train set
for a, b, _, _ in heldout_lme:
    HELD_PAIRS.append((a, b))
    HELD_PAIR_TOKS.append((toks(a), toks(b)))
n_lme = Counter()
for x in ku_train:
    for a, b in lme_pairs(x, True):
        for _ in range(2):  # small source: repeat
            add(nli_recs(a, b, "update", "lme/ku", "lme" + x["question_id"], both_orders=False))
        n_lme["update"] += 1
for x in ms_train:
    for a, b in lme_pairs(x, False):
        add(nli_recs(a, b, "compatible", "lme/ms", "lme" + x["question_id"], both_orders=False, with_relation=False))
        n_lme["compatible"] += 1

# ------------------------------------------------------------------ 3. PAWS (duplicate hard negatives)
paws = pd.read_parquet(snap("google-research-datasets/paws") / "labeled_final/train-00000-of-00001.parquet")
paws = paws.sample(frac=1.0, random_state=SEED)
pc = Counter()
for r in paws.itertuples():
    lab = int(r.label)
    if pc[lab] >= 450:
        continue
    pc[lab] += 1
    a, b = r.sentence1, r.sentence2
    if lab == 1:
        for x, y in ((a, b), (b, a)):
            s, q, o = KT.nli(x, y)
            add([rec(s, q, o, [KT.ENT], "paws", "duplicate", f"paws{r.id}", x, y)])
    else:  # not a paraphrase: anything but entailment
        s, q, o = KT.nli(a, b)
        add([rec(s, q, o, [KT.NEU, KT.CON], "paws", "not_duplicate", f"paws{r.id}", a, b)])

# ------------------------------------------------------------------ 4. MNLI (modest share; forgetting guard)
mdir = snap("nyu-mll/multi_nli") / "data"
mn = pd.read_parquet(mdir / "train-00000-of-00001.parquet").sample(n=6000, random_state=SEED)
mc = Counter()
MLAB = {0: KT.ENT, 1: KT.NEU, 2: KT.CON}
for r in mn.itertuples():
    if r.label not in MLAB or mc[r.label] >= 300:
        continue
    mc[r.label] += 1
    s, q, o = KT.nli(r.premise, r.hypothesis)
    add([rec(s, q, o, [MLAB[r.label]], "mnli", ["entailment", "neutral", "contradiction"][r.label],
             f"mnli{r.promptID}", r.premise, r.hypothesis)])
mdev = pd.read_parquet(mdir / "validation_matched-00000-of-00001.parquet").sample(n=600, random_state=SEED)
mnli_dev = [[r.premise, r.hypothesis, ["entailment", "neutral", "contradiction"][r.label]]
            for r in mdev.itertuples() if r.label in MLAB]

# ------------------------------------------------------------------ 5. programmatic memory pairs
mrng = random.Random(SEED + 2)
mem = memgen.memory_pairs(mrng, 950)
cls_count = Counter()
for a, b, cls, g in mem:
    add(nli_recs(a, b, cls, "mem", g, both_orders=True, with_relation=mrng.random() < 0.5))
    cls_count[cls] += 1
ent = memgen.entity_pairs(mrng, 650)
for a, b, lab, g in ent:
    for x, y in ((a, b), (b, a)):
        s, q, o = KT.entity(x, y)
        add([rec(s, q, o, [KT.SAME] if lab == "same" else [KT.DIFF], "mem/entity", lab, g, x, y)])
for st, q, opts, allowed, g in memgen.ontology_items(mrng, 600):
    records.append(rec(st, q, opts, allowed, "mem/ontology", "choice", g))

# ------------------------------------------------------------------ split + write
groups = sorted({r["group"] for r in records})
grng = random.Random(SEED + 3)
val_groups = set(grng.sample(groups, int(0.06 * len(groups))))
train = [r for r in records if r["group"] not in val_groups]
val = [r for r in records if r["group"] in val_groups]
grng.shuffle(train)
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "train.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT / "val.jsonl", "w") as f:
    for r in val:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
(OUT / "lme_ku_heldout.json").write_text(json.dumps(
    {"licence": "LongMemEval (xiaowu0162/longmemeval-cleaned), MIT",
     "note": "oracle file; knowledge-update pairs (positive, supersedes) vs multi-session pairs (negative, compatible) of held-out questions",
     "pairs": [[a, b, c] for a, b, c, _ in heldout_lme],
     "question_ids": sorted({q for *_, q in heldout_lme})}, ensure_ascii=False, indent=1))
(OUT / "mnli_dev.json").write_text(json.dumps(
    {"licence": "MultiNLI validation_matched (nyu-mll/multi_nli): OANC / CC BY-SA 3.0 / CC BY 3.0", "items": mnli_dev},
    ensure_ascii=False, indent=1))
manifest = {
    "seed": SEED,
    "train_requests": len(train), "val_requests": len(val),
    "by_source": Counter(r["src"] for r in train),
    "by_class": Counter(r["cls"] for r in train),
    "vitaminc_claim_evidence_per_label": per, "vitaminc_revision_pairs": n_rev,
    "lme_train_pairs": n_lme, "lme_heldout_pairs": Counter(c for _, _, c, _ in heldout_lme),
    "paws": pc, "mnli_train": mc, "mnli_dev": len(mnli_dev),
    "memgen_pairs": cls_count, "entity_pairs": len(ent),
    "dropped_near_heldout": report["dropped_near_heldout"],
    "licences": {
        "tals/vitaminc": "CC BY-SA 3.0 (Wikipedia-derived; share-alike)",
        "xiaowu0162/longmemeval-cleaned": "MIT",
        "google-research-datasets/paws": "PAWS licence: free for any use, attribution to Google LLC appreciated",
        "nyu-mll/multi_nli": "mixed per genre: OANC (free), CC BY-SA 3.0, CC BY 3.0",
        "programmatic": "kleos authors (MPL-2.0 with the repo)",
    },
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, default=dict))
print(json.dumps(manifest, indent=1, default=dict))
