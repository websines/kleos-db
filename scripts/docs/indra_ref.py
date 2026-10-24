"""Reference outputs of the Python Indra OCR package, for the Rust port's parity check.

    python indra_ref.py SAMPLES_DIR OUT.json [--mode auto|ocr] [--threads 4]

For every file in SAMPLES_DIR/manifest.json: per page the markdown, lines (box,
text, confidence), blocks, source, note, rotation and skew, and the wall time.
With --stages, OCR pages also record the detector boxes and the raw per-line
recognition (text, confidence), so a mismatch can be traced to its stage.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import indra_ocr


def main():
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "auto"
    threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4
    stages = "--stages" in sys.argv
    conv = indra_ocr.load(threads=threads, mode=mode)
    ocr = conv.ocr
    man = json.loads((root / "manifest.json").read_text())
    res = {}
    for item in man:
        path = root / item["file"]
        pages = []
        t0 = time.perf_counter()
        for i, page in conv.convert(path):
            d = {
                "index": i,
                "markdown": page.markdown,
                "lines": [{"box": ln.box, "text": ln.text, "conf": ln.conf} for ln in page.lines],
                "blocks": page.blocks,
                "source": page.source,
                "note": page.note,
                "rotation": page.rotation,
                "skew": page.skew,
                "timings": page.timings,
            }
            pages.append(d)
        dt = time.perf_counter() - t0
        if stages and path.suffix.lower() == ".png":
            import cv2

            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            boxes = ocr.detect(gray)
            rec = ocr.recognize(gray, boxes)
            pages[0]["stages"] = {
                "det": [[round(float(v), 3) for v in b] for b in boxes],
                "rec": [[t, round(c, 5)] for t, c, _ in rec],
            }
        res[item["file"]] = {"pages": pages, "seconds": dt}
        print(item["file"], len(pages), f"{dt:.2f}s", flush=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
