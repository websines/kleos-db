"""Generate the OCR parity and document-pipeline sample set.

Every page is made here (no third-party documents), so the set is free to use:
digital PDFs written with reportlab, and "scans" rasterized from them with
pdfium, then tilted, blurred and speckled with Pillow.  A few public-domain US
government PDFs can be added with ``--fetch`` (see FETCH; they are not needed).

    python make_samples.py OUT_DIR [--fetch]

Writes OUT_DIR/pdf/*.pdf, OUT_DIR/img/*.png and OUT_DIR/manifest.json.
"""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    ListFlowable,
    ListItem,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONTS = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("Uni", str(FONTS / "Arial Unicode.ttf")))
pdfmetrics.registerFont(TTFont("Serif", str(FONTS / "Times New Roman.ttf")))
pdfmetrics.registerFont(TTFont("Georgia", str(FONTS / "Georgia.ttf")))

# Public-domain works of the US federal government (17 U.S.C. 105).
FETCH = {
    "irs_fw4.pdf": "https://www.irs.gov/pub/irs-pdf/fw4.pdf",
}

PROSE = [
    "The harbour authority published its quarterly report on Tuesday. Cargo volumes rose by "
    "4.2 percent compared with the same period last year, driven mainly by container traffic "
    "from the northern routes. Bulk shipments of grain fell slightly after a dry summer.",
    "Maintenance of the eastern breakwater is scheduled to begin in March. The work will close "
    "berths 7 and 8 for about six weeks; vessels will be redirected to the western terminal, "
    "where two additional cranes were commissioned in January.",
    "Passenger numbers recovered strongly. Ferry services carried 1,284,000 passengers, the "
    "highest figure since records began, and the new evening sailing to Westport proved popular "
    "with commuters who previously relied on the coastal road.",
    "Environmental monitoring continued at all six stations. Average water temperature was "
    "14.8 degrees Celsius; dissolved oxygen remained within the permitted range throughout the "
    "quarter, and no exceedances of the turbidity limit were recorded.",
    "The board approved a revised tariff schedule that takes effect on the first of July. Dues "
    "for vessels under 5,000 gross tonnes are unchanged, while larger ships pay a surcharge of "
    "1.5 percent to fund the dredging programme in the approach channel.",
]

RUS = ("Отчёт о деятельности порта за третий квартал. Объём перевалки грузов вырос на четыре "
       "процента, а пассажиропоток достиг рекордного уровня. Ремонт восточного волнолома начнётся в марте.")
GRE = ("Η λιμενική αρχή δημοσίευσε την τριμηνιαία έκθεσή της. Ο όγκος φορτίων αυξήθηκε κατά "
       "τέσσερα τοις εκατό, ενώ η επιβατική κίνηση έφτασε σε ιστορικό υψηλό.")
CJK = ("港务局星期二发布了季度报告。与去年同期相比，货物吞吐量增长了百分之四点二，主要来自北方航线的集装箱运输。"
       "东防波堤的维修工程计划于三月开始，届时七号和八号泊位将关闭约六个星期。")


def styles():
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Serif", fontSize=11, leading=14)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Georgia", fontSize=18, leading=22)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Georgia", fontSize=14, leading=18)
    uni = ParagraphStyle("uni", parent=body, fontName="Uni")
    return body, h1, h2, uni


def prose_pdf(path: Path, seed: int):
    body, h1, h2, _ = styles()
    rnd = random.Random(seed)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm)
    story = [Paragraph("Quarterly Report of the Harbour Authority", h1), Spacer(1, 8)]
    for k in range(2):
        story.append(Paragraph(f"{k + 1}. {['Operations', 'Finance'][k]}", h2))
        for p in rnd.sample(PROSE, 3):
            story += [Paragraph(p, body), Spacer(1, 6)]
    story.append(Paragraph("Key points", h2))
    story.append(
        ListFlowable(
            [ListItem(Paragraph(t, body)) for t in [
                "Cargo volumes rose by 4.2 percent.",
                "Berths 7 and 8 close for six weeks from March.",
                "A revised tariff schedule starts on 1 July.",
            ]],
            bulletType="bullet",
        )
    )
    doc.build(story)


def two_column_pdf(path: Path):
    body, h1, h2, _ = styles()
    doc = BaseDocTemplate(str(path), pagesize=letter)
    w, h = letter
    gap = 0.8 * cm
    fw = (w - 2 * 2 * cm - gap) / 2
    frames = [Frame(2 * cm, 2 * cm, fw, h - 4 * cm, id="c1"), Frame(2 * cm + fw + gap, 2 * cm, fw, h - 4 * cm, id="c2")]
    doc.addPageTemplates([PageTemplate(id="two", frames=frames)])
    story = [Paragraph("Notes on Coastal Sediment Transport", h1)]
    for i in range(3):
        story.append(Paragraph(f"Section {i + 1}", h2))
        for p in PROSE:
            story += [Paragraph(p, body), Spacer(1, 4)]
    doc.build(story)


def table_pdf(path: Path):
    body, h1, h2, _ = styles()
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    data = [["Terminal", "Berths", "Cranes", "Throughput (TEU)", "Change"]]
    rnd = random.Random(7)
    for name in ["North", "East", "West", "South", "Inner Basin", "Ferry Quay", "Grain Pier", "Oil Jetty"]:
        data.append([name, str(rnd.randint(2, 12)), str(rnd.randint(0, 9)), f"{rnd.randint(10, 900) * 1000:,}", f"{rnd.uniform(-9, 9):+.1f}%"])
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Serif"),
        ("FONTNAME", (0, 0), (-1, 0), "Georgia"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story = [Paragraph("Terminal statistics", h1), Paragraph(PROSE[0], body), Spacer(1, 10), t, Spacer(1, 10), Paragraph(PROSE[4], body)]
    doc.build(story)


def unicode_pdf(path: Path, text: str, title: str):
    body, h1, h2, uni = styles()
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    story = [Paragraph(title, ParagraphStyle("t", parent=uni, fontSize=18, leading=22)), Spacer(1, 10)]
    for _ in range(4):
        story += [Paragraph(text, uni), Spacer(1, 8)]
    doc.build(story)


def scan_of(pdf: Path, out_png: Path, seed: int, dpi: float = 200.0, tilt: float = 0.0):
    import pypdfium2 as pdfium
    from PIL import Image, ImageFilter

    rnd = random.Random(seed)
    doc = pdfium.PdfDocument(str(pdf))
    img = doc[0].render(scale=dpi / 72.0, grayscale=True).to_pil().convert("L")
    doc.close()
    if tilt:
        img = img.rotate(tilt, resample=Image.BICUBIC, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    px = img.load()
    w, h = img.size
    for _ in range(w * h // 400):  # speckle
        x, y = rnd.randrange(w), rnd.randrange(h)
        px[x, y] = rnd.choice((0, 60, 200, 255))
    img = img.point(lambda v: min(255, max(0, int(v * 0.92 + 14))))  # grey paper
    img.save(out_png)
    return out_png


def image_pdf(png: Path, out: Path):
    from PIL import Image

    Image.open(png).convert("L").save(out, "PDF", resolution=200.0)


def main():
    out = Path(sys.argv[1])
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(parents=True, exist_ok=True)
    man = []

    def add(p, kind, note):
        man.append({"file": str(p.relative_to(out)), "kind": kind, "note": note, "source": "generated (make_samples.py)"})

    for s in range(4):
        p = out / "pdf" / f"prose_{s}.pdf"
        prose_pdf(p, s)
        add(p, "digital", "single column prose, headings, bullet list")
    p = out / "pdf" / "two_column.pdf"
    two_column_pdf(p)
    add(p, "digital", "two columns")
    p = out / "pdf" / "table.pdf"
    table_pdf(p)
    add(p, "digital", "ruled table")
    for name, text, title in [("cyrillic", RUS, "Отчёт порта"), ("greek", GRE, "Έκθεση λιμένα"), ("cjk", CJK, "港务局季度报告")]:
        p = out / "pdf" / f"{name}.pdf"
        unicode_pdf(p, text, title)
        add(p, "digital", f"non-Latin: {name}")
    # scans: images and image-only PDFs (no text layer)
    scans = [("prose_0", 0.0), ("prose_1", 1.2), ("two_column", 0.0), ("table", -0.8), ("cyrillic", 0.0), ("cjk", 0.0), ("prose_2", 2.5), ("greek", 0.5)]
    for i, (src, tilt) in enumerate(scans):
        png = scan_of(out / "pdf" / f"{src}.pdf", out / "img" / f"scan_{src}.png", i, tilt=tilt)
        add(png, "scan", f"raster of {src}.pdf at 200 dpi, tilt {tilt} deg, blur, speckle")
        if i < 4:
            p = out / "pdf" / f"scan_{src}.pdf"
            image_pdf(png, p)
            add(p, "scan-pdf", f"image-only PDF of scan_{src}.png")
    if "--fetch" in sys.argv:
        for name, url in FETCH.items():
            p = out / "pdf" / name
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "kleos-samples"})
                p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
                man.append({"file": str(p.relative_to(out)), "kind": "digital", "note": "US government form", "source": url + " (public domain, 17 U.S.C. 105)"})
            except Exception as e:  # noqa: BLE001
                print("fetch failed", name, e)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(len(man), "files")


if __name__ == "__main__":
    main()
