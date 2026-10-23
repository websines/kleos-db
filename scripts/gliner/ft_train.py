# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Fine-tune GLiNER2.5 on the silver-labelled pool (ft_data.py).

One multi-task example per claim: the four entity types (bare label names,
an empty type trained as a negative), `kind` (7 classes) and `about_speaker`
(yes/no). Silver labels are cleaned first (clean()). 10% of the pool is held
out as validation (threshold tuning in ft_eval.py). gliner2's trainer knows
CUDA and CPU only; MPS is patched in, with the token table frozen.

    python ft_train.py build                       # train.jsonl / val.jsonl
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
    python ft_train.py train base 3 base-e3        # -> $GLINER_DIR/ckpt/base-e3/final
"""
import json
import os
import random
import sys
import time

os.environ.setdefault("GLINER_DIR", "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/gliner-ft")
import common as C  # noqa: E402
from ft_data import ENTS, KINDS  # noqa: E402

ENT_DESC = {
    "person": "name of a specific person or pet",
    "location": "name of a specific place: city, country, region, venue, park",
    "organization": "name of a company, employer, school, team, website or app",
    "relative": "phrase for one specific person by relation or role to the speaker: my sister, Mom, my boss",
}
KIND_DESC = {
    "fact": "something stable or current about the speaker, someone or the world",
    "preference": "a like, dislike, favourite or opinion",
    "plan": "an intention, plan, goal or future arrangement",
    "event": "something that happened at a particular time",
    "question": "asks for information",
    "request": "asks the listener to do or produce something",
    "small_talk": "greeting, thanks, reaction or compliment with no information",
}
ABOUT = ["yes", "no"]
ABOUT_PROMPT = "says something about the speaker"


def example(x, described=False):
    """Fine-tuning uses bare label names (no descriptions): the model learns
    what the labels mean, and the schema, which is encoded with every claim
    and dominates the cost (docs/research/gliner.md, Speed), is ~3x shorter."""
    from gliner2.training import Classification, InputExample
    ents = {k: [] for k in ENTS}
    for e in x["entities"]:
        if e["text"] not in ents[e["label"]]:
            ents[e["label"]].append(e["text"])
    return InputExample(
        text=x["text"], entities=ents, entity_descriptions=ENT_DESC if described else None,
        classifications=[
            Classification(task="kind", labels=KINDS, true_label=x["kind"], label_descriptions=KIND_DESC if described else None),
            Classification(task="about_speaker", labels=ABOUT, true_label="yes" if x["about_speaker"] else "no"),
        ])


PRONOUNS = {"i", "me", "my", "mine", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "they", "them", "their", "it"}


def clean(x):
    """Silver-label cleanup: a person, place or organization name has a
    capital letter (the no-reasoning labels tag "books" as a person), no
    pronoun is an entity ("I" as a person), and every mention is a whole-word
    match (gliner2 cannot align "I" inside "It")."""
    import re
    ents = [e for e in x["entities"]
            if (e["label"] == "relative" or any(c.isupper() for c in e["text"]))
            and e["text"].lower() not in PRONOUNS
            and re.search(r"(?<![\w-])" + re.escape(e["text"]) + r"(?![\w-])", x["text"])]
    return {**x, "entities": ents, "dropped": len(x["entities"]) - len(ents)}


def build():
    items = []
    for name in ("pool", "synth", "synth_chat"):
        p = C.GDIR / f"{name}_silver.json"
        if p.exists():
            items += [clean(x) for x in json.loads(p.read_text()) if "error" not in x]
    print(sum(x["dropped"] for x in items), "lower-case names dropped from", sum(len(x["entities"]) + x["dropped"] for x in items), "silver entities")
    rnd = random.Random(C.SEED + 13)
    rnd.shuffle(items)
    nv = len(items) // 10
    val, train = items[:nv], items[nv:]
    for name, xs in (("train", train), ("val", val)):
        from gliner2.training import TrainingDataset
        ds = TrainingDataset([example(x) for x in xs])
        ds.validate()
        ds.save(str(C.GDIR / f"{name}.jsonl"))
        (C.GDIR / f"{name}_items.json").write_text(json.dumps(xs, indent=1))
    print(len(train), "train,", len(val), "val")


def train(model="base", epochs=3, tag=None, batch=4, accum=2):
    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    class MPSTrainer(ExtractorTrainer):
        def _setup_device(self):
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.is_distributed = False
            self.model.to(self.device)
            self.model.float()

        def _optimizer_step(self):
            ok = super()._optimizer_step()
            if self.device.type == "mps" and self.global_step % 25 == 0:
                torch.mps.empty_cache()  # the MPS allocator's cache otherwise grows to 10+ GB on a 16 GB machine
            return ok

    tag = tag or f"{model}-e{epochs}"
    out = C.GDIR / "ckpt" / tag
    m = AutoExtractor.from_pretrained(C.MODELS[model])
    # the 128k-row token table is half of base's parameters; freezing it
    # halves the optimizer state and keeps training inside ~7 GB of MPS memory
    for n, p in m.named_parameters():
        if "word_embeddings" in n:
            p.requires_grad = False
    cfg = TrainingConfig(
        output_dir=str(out), experiment_name=tag, num_epochs=epochs, batch_size=batch, gradient_accumulation_steps=accum, eval_batch_size=8,
        encoder_lr=1e-5, task_lr=5e-5 if model != "small" else 1e-4, warmup_ratio=0.1, scheduler_type="linear",
        fp16=False, bf16=False, eval_strategy="epoch", save_best=True, metric_for_best="eval_loss",
        save_total_limit=1, logging_steps=10, num_workers=0, pin_memory=False, seed=C.SEED,
        validate_data=True, on_capacity_exceeded="truncate_with_warning",
    )
    t0 = time.time()
    tr = MPSTrainer(m, cfg)
    res = tr.train(str(C.GDIR / "train.jsonl"), str(C.GDIR / "val.jsonl"))
    secs = time.time() - t0
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    (out / "train_summary.json").write_text(json.dumps({"model": model, "epochs": epochs, "secs": secs, "peak_rss_gb": rss,
                                                        "result": {k: v for k, v in res.items() if isinstance(v, (int, float, str))}}, indent=1, default=str))
    print(f"{tag}: {secs / 60:.1f} min, peak RSS {rss:.2f} GB")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4, sys.argv[4] if len(sys.argv) > 4 else None)
