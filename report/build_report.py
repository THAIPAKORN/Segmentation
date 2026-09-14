"""
สร้าง report/index.html จาก template.html

ทำ 2 อย่าง
  1. ย่อรูปจาก results/figures ให้เล็กลง แล้วฝังลงไปในไฟล์ HTML เลย
     (จะได้เป็นไฟล์เดียวจบ ส่งต่อได้ ไม่ต้องแนบโฟลเดอร์รูป)
  2. วาดรูป Structuring Element ขนาดต่าง ๆ จาก kernel จริงของ OpenCV
"""
import base64
import io
from pathlib import Path

import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "results" / "figures"

# ชื่อที่ใช้ใน template.html -> ชื่อไฟล์รูปจริง
PICTURES = {
    "CONFUSION": "รูป1_confusion_matrix.png",
    "ROC": "รูป2_roc.png",
    "SESIZE": "รูป3_ขนาด_SE.png",
    "STEPS": "รูป4_ขั้นตอน.png",
    "EXAMPLES": "รูป5_ตัวอย่าง.png",
}
PHOTO = {"รูป4_ขั้นตอน.png", "รูป5_ตัวอย่าง.png"}   # รูปที่มีภาพถ่าย ใช้ JPEG จะเล็กกว่า


def shrink_and_embed(path):
    """ย่อรูปแล้วแปลงเป็นข้อความ base64 สำหรับฝังใน HTML"""
    image = Image.open(path).convert("RGB")
    if image.width > 1250:
        image = image.resize((1250, round(image.height * 1250 / image.width)),
                             Image.LANCZOS)
    buffer = io.BytesIO()
    if path.name in PHOTO:
        image.save(buffer, "JPEG", quality=82, optimize=True)
        kind = "jpeg"
    else:
        image.save(buffer, "PNG", optimize=True)
        kind = "png"
    text = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{kind};base64,{text}"


def draw_se(size, box=96):
    """วาด Structuring Element รูปวงรีขนาดที่กำหนด เป็นภาพ SVG"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    cell = box / size
    gap = max(cell * 0.09, 0.35)
    squares = []
    for y in range(size):
        for x in range(size):
            color = "var(--grid-on)" if kernel[y, x] else "var(--grid-off)"
            squares.append(
                f'<rect x="{x * cell + gap / 2:.2f}" y="{y * cell + gap / 2:.2f}" '
                f'width="{cell - gap:.2f}" height="{cell - gap:.2f}" fill="{color}"/>')
    return (f'<svg width="{box}" height="{box}" viewBox="0 0 {box} {box}" '
            f'role="img" aria-label="structuring element วงรี {size} คูณ {size}">'
            + "".join(squares) + "</svg>")


def se_gallery():
    blocks = []
    for size in (5, 13, 21, 41):
        chosen = " sel" if size == 21 else ""
        note = " ← เลือกใช้" if chosen else ""
        blocks.append(f'<div class="se{chosen}">{draw_se(size)}'
                      f'<div class="lab">{size}×{size}{note}</div></div>')
    return "".join(blocks)


def main():
    html = (HERE / "template.html").read_text(encoding="utf-8")
    for key, filename in PICTURES.items():
        html = html.replace("{{" + key + "}}", shrink_and_embed(FIGURES / filename))
    html = html.replace("{{SEGRIDS}}", se_gallery())

    assert "{{" not in html, "ยังมีช่องว่างที่ยังไม่ได้แทนที่"
    out = HERE / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"สร้างเสร็จ -> {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
