"""Central paths and experiment settings for the Segmentation workshop."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMG_DIR = DATA / "images"          # ภาพต้นฉบับที่คัดมาแล้ว (RGB)
GT_DIR = DATA / "gt"               # ground truth binary mask (0 / 255)
VALID_DIR = DATA / "valid"         # pixel ที่นำมาคิดคะแนน (ตัดขอบ boundary ทิ้ง)
META_CSV = DATA / "metadata.csv"

RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
TAB_DIR = RESULTS / "tables"
PRED_DIR = RESULTS / "predictions"  # score map + binary mask ของแต่ละวิธี

RAW = Path(
    r"C:\Users\Pakorn\AppData\Local\Temp\claude"
    r"\C--Learn-Uni-Y3-Term1-Image-processing-Segmentation"
    r"\06c51913-119a-4a2e-9152-78ee587a3612\scratchpad\raw"
)
RAW_IMAGES_TGZ = RAW / "images.tar.gz"
TRIMAP_DIR = RAW / "annotations" / "trimaps"
LIST_TXT = RAW / "annotations" / "list.txt"

# ---------------------------------------------------------------- dataset
PER_BREED = 2        # 37 breeds x 2 = 74 ภาพ (โจทย์กำหนดอย่างน้อย 50)
MAX_SIDE = 400       # ย่อภาพให้ด้านยาวสุด = 400 px เพื่อความเร็ว
SEED = 2026

# Oxford-IIIT Pet trimap encoding
TRIMAP_FG = 1        # ตัวสัตว์      -> Positive class
TRIMAP_BG = 2        # พื้นหลัง      -> Negative class
TRIMAP_BOUNDARY = 3  # เส้นขอบกำกวม -> ไม่นำมาคิด (don't care)

# ---------------------------------------------------------------- method
BORDER_FRAC = 0.08   # สัดส่วนความกว้างขอบภาพที่ถือว่าเป็น "พื้นหลังแน่ ๆ"
GMM_K = 4            # จำนวน component ของ background GMM
CENTER_SIGMA = 0.40  # sigma ของ center prior (หน่วย = สัดส่วนของขนาดภาพ)

# ---------------------------------------------------------------- morphology
# ค่าเหล่านี้เลือกจากชุด TUNE (37 ภาพ) ด้วย 03_morphology_study.py
# และยืนยันแล้วว่าช่วยจริงบนชุด TEST (37 ภาพ) ที่ไม่เคยใช้เลือกค่า
SE_SHAPE = "ellipse"
SE_SIZE = 21
MORPH_OPEN = False        # Opening ทำให้แย่ลงในโจทย์นี้ (ลบขา/หาง/หูที่บางออก)
MORPH_CLOSE = True        # Closing ช่วยมากที่สุด (เชื่อมตัวสัตว์ที่ขาดเป็นชิ้น)
MORPH_FILL = True         # Region filling อุดรูภายในตัวสัตว์
MORPH_LARGEST_CC = False  # เก็บเฉพาะก้อนใหญ่สุด -> เสียส่วนที่ขาดจากกัน

for _d in (IMG_DIR, GT_DIR, VALID_DIR, FIG_DIR, TAB_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)
