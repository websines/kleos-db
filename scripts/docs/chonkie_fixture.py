"""Chonkie reference outputs for kleos-docs' chunk parity test.

    python chonkie_fixture.py RUST_PAGES.json OUT.json

Texts: the Markdown of a few sample pages (generated documents, see
make_samples.py) plus a Markdown/code sample. For each text: Chonkie 1.7's
TokenChunker, SentenceChunker and RecursiveChunker with the character and
word tokenizers, at two sizes.
"""

import json
import sys

from chonkie import RecursiveChunker, SentenceChunker, TokenChunker

pages = json.load(open(sys.argv[1]))
texts = []
for f in ["pdf/prose_0.pdf", "pdf/two_column.pdf", "pdf/table.pdf", "pdf/cjk.pdf", "img/scan_cyrillic.png"]:
    texts.append(pages[f]["pages"][0]["markdown"])
texts.append("# Title\n\nSome intro, with commas; and pauses: here.\n\n```python\ndef f(x):\n    return x + 1\n```\n\n- item one\n- item two... and more\n\nA last paragraph (with brackets) that runs on for a while to force splits.\n")
out = []
for t in texts:
    for tok in ["character", "word"]:
        for size in ([200, 600] if tok == "character" else [40, 120]):
            rec = {"text": t, "tokenizer": tok, "size": size}
            rec["token"] = [[c.text, c.start_index, c.end_index, c.token_count] for c in TokenChunker(tok, chunk_size=size, chunk_overlap=size // 10)(t)]
            rec["sentence"] = [[c.text, c.start_index, c.end_index, c.token_count] for c in SentenceChunker(tok, chunk_size=size, chunk_overlap=0)(t)]
            rec["recursive"] = [[c.text, c.start_index, c.end_index, c.token_count] for c in RecursiveChunker(tok, chunk_size=size)(t)]
            out.append(rec)
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
print(len(out), "cases")
