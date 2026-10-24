"""Python Indra OCR on selected pages of one PDF (reference for the Rust port).

    python indra_pages.py BOOK.pdf OUT.json PAGES [--threads 4] [--workers 4] [--all]

--workers is passed as convert_many's `cores`.

PAGES: comma-separated 0-based page numbers. Writes per page the line texts,
the Markdown and the time. With --all, also converts the whole book with
`convert_many` on `--workers` processes and reports pages per second.
"""

import json
import sys
import time

import indra_ocr
import pypdfium2 as pdfium


def main():
    book, out = sys.argv[1], sys.argv[2]
    pages = [int(x) for x in sys.argv[3].split(",") if x]
    threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4
    t0 = time.time()
    conv = indra_ocr.load(threads=threads)
    load_s = time.time() - t0
    doc = pdfium.PdfDocument(book)
    res = {"book": book, "load_s": round(load_s, 2), "threads": threads, "pages": {}}
    t0 = time.time()
    for i in pages:
        t = time.time()
        p = conv.pdf_page(book, doc, i)
        res["pages"][str(i)] = {"lines": [l.text for l in p.lines], "markdown": p.markdown, "source": p.source, "s": round(time.time() - t, 3)}
    res["sample_s"] = round(time.time() - t0, 2)
    res["sample_pages_per_s"] = round(len(pages) / (time.time() - t0), 3)
    doc.close()
    if "--all" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
        t0 = time.time()
        n = 0
        for _path, result in indra_ocr.convert_many([book], cores=workers):
            n += len(result) if isinstance(result, list) else 0
        res["all_pages"] = n
        res["all_s"] = round(time.time() - t0, 1)
        res["all_pages_per_s"] = round(n / (time.time() - t0), 3)
        res["all_workers"] = workers
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print({k: v for k, v in res.items() if k != "pages"})


if __name__ == "__main__":
    main()
