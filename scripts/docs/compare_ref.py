"""Parity of the Rust Indra port against the Python package.

    python compare_ref.py REF.json RUST.json [--md]

Per page: character F1 of the page text (lines joined in reading order; with
--md the Markdown), exact-text share of lines, block-label agreement (the
label sequences aligned), and the source chosen (text / ocr / hybrid).  Prints
a table and the totals; character F1 is 2 * matched / (len_a + len_b) with
difflib's matching blocks, so 1.0 is identical text.
"""

from __future__ import annotations

import difflib
import json
import sys


def char_f1(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    m = sum(x.size for x in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return 2 * m / (len(a) + len(b))


def label_agree(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    m = sum(x.size for x in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return 2 * m / (len(a) + len(b))


def main():
    ref = json.load(open(sys.argv[1]))
    rust = json.load(open(sys.argv[2]))
    md = "--md" in sys.argv
    rows = []
    for f, r in ref.items():
        if f not in rust:
            continue
        for pr, pu in zip(r["pages"], rust[f]["pages"]):
            ta = (pr["markdown"] or "") if md else "\n".join(l["text"] for l in pr["lines"])
            tb = (pu["markdown"] or "") if md else "\n".join(l["text"] for l in pu["lines"])
            la = [l["text"] for l in pr["lines"]]
            lb = set(l["text"] for l in pu["lines"])
            exact = sum(t in lb for t in la) / max(len(la), 1)
            rows.append({
                "page": f"{f}#{pr['index']}",
                "f1": char_f1(ta, tb),
                "exact": exact,
                "blocks": label_agree([b["label"] for b in pr["blocks"]], [b["label"] for b in pu["blocks"]]),
                "src": f"{pr['source']}/{pu['source']}",
                "n": (len(pr["lines"]), len(pu["lines"])),
            })
    print(f"{'page':38s} {'charF1':>7s} {'exact':>6s} {'blocks':>6s}  src(py/rs)  lines(py/rs)")
    for x in rows:
        print(f"{x['page']:38s} {x['f1']:7.4f} {x['exact']:6.3f} {x['blocks']:6.3f}  {x['src']:11s} {x['n']}")
    n = len(rows)
    print(f"\n{n} pages: mean char F1 {sum(x['f1'] for x in rows) / n:.4f}, min {min(x['f1'] for x in rows):.4f}; "
          f"lines identical {sum(x['exact'] for x in rows) / n:.3f}; block labels {sum(x['blocks'] for x in rows) / n:.3f}; "
          f"same source {sum(x['src'].split('/')[0] == x['src'].split('/')[1] for x in rows)}/{n}")
    ts = [x for x in rows]
    for kind in ("text", "ocr", "hybrid"):
        k = [x for x in ts if x["src"].split("/")[0] == kind]
        if k:
            print(f"  {kind:6s}: {len(k)} pages, char F1 {sum(x['f1'] for x in k) / len(k):.4f}")


if __name__ == "__main__":
    main()
