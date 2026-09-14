"""
หัวใจของงานทั้งหมดอยู่ในไฟล์นี้ อ่านไฟล์เดียวก็เข้าใจว่าเราทำอะไร

โจทย์ : ภาพหมา 1 ตัว บนพื้นหลัง -> ระบายว่า pixel ไหนคือ "หมา" pixel ไหนคือ "พื้นหลัง"

ไอเดีย  ขอบภาพรอบนอกเกือบทั้งหมดคือพื้นหลัง
        เอาสีตรงขอบมาเฉลี่ย ก็จะรู้ว่า "สีพื้นหลังหน้าตาประมาณไหน"
        pixel ไหนสีต่างจากพื้นหลังมาก ก็น่าจะเป็นตัวหมา

ขั้นตอน 1. หาสีเฉลี่ยของขอบภาพ            = สีพื้นหลัง
        2. วัดว่าแต่ละ pixel สีต่างจากพื้นหลังเท่าไร = ได้ "ภาพคะแนน"
        3. ตัดภาพคะแนนเป็นขาว-ดำ ด้วย Otsu     = ได้คำตอบดิบ ๆ
        4. เก็บกวาดด้วย Morphology              = ได้คำตอบสุดท้าย
"""
import cv2
import numpy as np
from scipy import ndimage

# ----- ค่าที่ปรับได้ -----
BORDER = 0.10   # ใช้ขอบภาพรอบนอก 10% เป็นตัวอย่างของพื้นหลัง
SE_SIZE = 21    # ขนาดของ Structuring Element ที่ใช้ทำ Morphology


def background_color(image):
    """ขั้นที่ 1 : สีเฉลี่ยของขอบภาพ = สีพื้นหลัง"""
    h, w = image.shape[:2]
    bh, bw = int(h * BORDER), int(w * BORDER)
    edge = np.concatenate([
        image[:bh].reshape(-1, 3),      # แถบบน
        image[-bh:].reshape(-1, 3),     # แถบล่าง
        image[:, :bw].reshape(-1, 3),   # แถบซ้าย
        image[:, -bw:].reshape(-1, 3),  # แถบขวา
    ])
    return edge.mean(axis=0)


def make_score(image_bgr):
    """
    ขั้นที่ 2 : สร้าง "ภาพคะแนน" ค่า 0-255
    ยิ่งสว่าง = สีต่างจากพื้นหลังมาก = ยิ่งน่าจะเป็นตัวหมา
    """
    blur = cv2.medianBlur(image_bgr, 5)                      # ลด noise ก่อน
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(float)  # ใช้ปริภูมิสี Lab

    bg = background_color(lab)
    distance = np.sqrt(((lab - bg) ** 2).sum(axis=2))        # ระยะห่างเชิงสี

    return (distance / distance.max() * 255).astype(np.uint8)


def to_black_white(score):
    """ขั้นที่ 3 : ตัดภาพคะแนนเป็นขาว-ดำ ด้วย Otsu (True = หมา)"""
    _, mask = cv2.threshold(score, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def clean_with_morphology(mask, se_size=SE_SIZE):
    """
    ขั้นที่ 4 : เก็บกวาดด้วย Morphology

    Closing (Dilation แล้ว Erosion) : เชื่อมตัวหมาที่ขาดเป็นชิ้น ๆ ให้ติดกัน
    Fill holes                      : อุดรูโหว่ที่อยู่ข้างในตัวหมา
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, se)
    return ndimage.binary_fill_holes(closed > 0)


def segment(image_bgr, use_morphology=True):
    """รวมทุกขั้นเข้าด้วยกัน -> คืน (ภาพคะแนน, คำตอบขาว-ดำ)"""
    score = make_score(image_bgr)
    mask = to_black_white(score)
    if use_morphology:
        mask = clean_with_morphology(mask)
    return score, mask


# ---------------------------------------------------------------- การวัดผล
def count_pixels(predict, truth, use_pixel):
    """
    นับ 4 ช่องของ Confusion Matrix
      TP = ทายว่าหมา และเป็นหมาจริง        FP = ทายว่าหมา แต่เป็นพื้นหลัง
      FN = ทายว่าพื้นหลัง แต่เป็นหมา        TN = ทายว่าพื้นหลัง และเป็นพื้นหลังจริง
    use_pixel = pixel ที่นำมานับ (ตัดแถบขอบที่ ground truth เองก็ไม่แน่ใจออก)
    """
    p, t = predict[use_pixel], truth[use_pixel]
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(~p & ~t))
    return tp, fp, fn, tn


def measures(tp, fp, fn, tn):
    """คำนวณค่าต่าง ๆ จาก Confusion Matrix (ตามสูตรในสไลด์)"""
    small = 1e-9
    precision = tp / (tp + fp + small)
    recall = tp / (tp + fn + small)          # = TPR = Hit rate
    return {
        "accuracy": (tp + tn) / (tp + fp + fn + tn + small),
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp + small),               # = TNR
        "fpr": fp / (fp + tn + small),                       # = False alarm
        "f1": 2 * precision * recall / (precision + recall + small),
        "iou": tp / (tp + fp + fn + small),
    }
