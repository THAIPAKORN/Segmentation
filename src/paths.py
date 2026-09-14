"""ที่อยู่ของโฟลเดอร์ทั้งหมด รวมไว้ที่เดียวจะได้แก้ง่าย"""
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# ---- ภาพที่ใช้ทดลอง ----
IMAGES_DIR = PROJECT / "data" / "images"   # ภาพหมา
TRUTH_DIR = PROJECT / "data" / "truth"     # คำตอบที่ถูกต้อง
USE_DIR = PROJECT / "data" / "use"         # pixel ที่เอามานับคะแนน

# ---- ผลลัพธ์ ----
FIGURE_DIR = PROJECT / "results" / "figures"
TABLE_DIR = PROJECT / "results" / "tables"

# ---- ไฟล์ดิบที่ดาวน์โหลดมาจาก Oxford (ใช้เฉพาะตอนเตรียมข้อมูล) ----
RAW = Path(
    r"C:\Users\Pakorn\AppData\Local\Temp\claude"
    r"\C--Learn-Uni-Y3-Term1-Image-processing-Segmentation"
    r"\06c51913-119a-4a2e-9152-78ee587a3612\scratchpad\raw"
)
IMAGES_TARGZ = RAW / "images.tar.gz"
ANNOTATION_DIR = RAW / "annotations" / "trimaps"


def clear_folder(folder):
    """สร้างโฟลเดอร์ถ้ายังไม่มี และลบไฟล์เก่าทิ้ง"""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*"):
        if old.is_file():
            old.unlink()


def image_names():
    """ชื่อภาพทั้งหมดที่เตรียมไว้แล้ว"""
    return sorted(p.stem for p in IMAGES_DIR.glob("*.jpg"))
