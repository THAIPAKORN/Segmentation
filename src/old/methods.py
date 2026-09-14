"""
Workshop ข้อ 3 + ข้อ 4 : Algorithm แยกวัตถุ (Positive) ออกจากพื้นหลัง (Negative)

แนวคิด : ทุกวิธีคืนค่าเป็น "score map" ค่า 0-1 ต่อ pixel (ยิ่งสูง = ยิ่งน่าจะเป็นวัตถุ)
         การมี score map ต่อเนื่องทำให้เขียน ROC curve ได้ (โจทย์ข้อ 5)
         ส่วน binary mask ได้จากการ threshold score map ด้วย Otsu

M1 Baseline    : ระยะห่างของความสว่าง (grayscale) จากค่าเฉลี่ยขอบภาพ
M2 Color-GMM   : ระยะห่างเชิงสีใน CIE-Lab จากโมเดลพื้นหลัง (GMM ที่ fit จากขอบภาพ)
M3 + center    : M2 คูณด้วย center prior (วัตถุมักอยู่กลางภาพ)
M4 = M3 + Morphology post-processing  <-- โจทย์ข้อ 4
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture

import config as C

SE_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


# --------------------------------------------------------------- utilities
def normalize(x, lo=1.0, hi=99.0):
    """ยืด score ให้อยู่ช่วง 0-1 โดยตัด outlier ด้วย percentile"""
    a, b = np.percentile(x, [lo, hi])
    if b - a < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - a) / (b - a), 0, 1).astype(np.float32)


def border_mask(shape, frac=C.BORDER_FRAC):
    """mask กรอบรอบนอกของภาพ = ตัวอย่างพื้นหลังที่เชื่อถือได้"""
    h, w = shape[:2]
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    m = np.zeros((h, w), bool)
    m[:bh, :] = m[-bh:, :] = m[:, :bw] = m[:, -bw:] = True
    return m


def center_prior(shape, sigma=C.CENTER_SIGMA):
    """ค่าถ่วงน้ำหนักรูประฆังคว่ำ สูงสุดที่กลางภาพ"""
    h, w = shape[:2]
    yy = (np.arange(h) - (h - 1) / 2) / (sigma * h)
    xx = (np.arange(w) - (w - 1) / 2) / (sigma * w)
    return np.exp(-0.5 * (yy[:, None] ** 2 + xx[None, :] ** 2)).astype(np.float32)


def structuring_element(shape=C.SE_SHAPE, size=C.SE_SIZE):
    """Structuring Element ตามนิยามใน Lecture 10"""
    return cv2.getStructuringElement(SE_SHAPES[shape], (size, size))


# --------------------------------------------------------------- score maps
def score_m1_gray(img_bgr):
    """M1 : |I - mean(I ที่ขอบภาพ)| บน grayscale ที่ผ่าน median filter"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5).astype(np.float32)
    mu = gray[border_mask(gray.shape)].mean()
    return normalize(np.abs(gray - mu))


def _bg_gmm_nll(img_bgr):
    """-log likelihood ของแต่ละ pixel ภายใต้ GMM ของสีพื้นหลัง (CIE-Lab)"""
    smooth = cv2.bilateralFilter(img_bgr, 9, 60, 15)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)

    bg = lab[border_mask(lab.shape)].reshape(-1, 3)
    if len(bg) > 6000:                       # สุ่มลดจำนวนเพื่อความเร็ว
        idx = np.random.default_rng(C.SEED).choice(len(bg), 6000, replace=False)
        bg = bg[idx]

    gmm = GaussianMixture(C.GMM_K, covariance_type="full", random_state=C.SEED,
                          reg_covar=1e-3).fit(bg)
    return (-gmm.score_samples(flat)).reshape(lab.shape[:2])


def score_m2_color(img_bgr):
    """M2 : ความ 'ไม่เหมือนพื้นหลัง' เชิงสี"""
    return normalize(_bg_gmm_nll(img_bgr))


def score_m3_color_center(img_bgr):
    """M3 : M2 x center prior  <-- วิธีหลักที่เสนอ"""
    return normalize(score_m2_color(img_bgr) * center_prior(img_bgr.shape))


# --------------------------------------------------------------- threshold
def otsu(score01):
    """Otsu threshold บน score map -> คืนค่า (mask bool, threshold ในหน่วย 0-1)"""
    u8 = np.round(score01 * 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return score01 > (t / 255.0), t / 255.0


# --------------------------------------------------------------- morphology
def postprocess(mask, se_shape=C.SE_SHAPE, se_size=C.SE_SIZE,
                do_open=C.MORPH_OPEN, do_close=C.MORPH_CLOSE,
                fill=C.MORPH_FILL, largest_cc=C.MORPH_LARGEST_CC):
    """
    โจทย์ข้อ 4 : ปรับปรุงผล segmentation ด้วย Morphology
      Opening  = Erosion -> Dilation : ลบจุดรบกวนเล็ก ๆ ในพื้นหลัง / ตัดติ่งบาง ๆ
      Closing  = Dilation -> Erosion : เชื่อมรอยขาด / อุดรูเล็ก ๆ ในตัววัตถุ
      Fill holes                     : อุดรูที่ปิดล้อมสนิท (region filling)
      Largest CC                     : เก็บเฉพาะ connected component ที่ใหญ่ที่สุด
    """
    m = mask.astype(np.uint8)
    if se_size >= 3 and (do_open or do_close):
        se = structuring_element(se_shape, se_size)
        if do_open:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
        if do_close:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
    m = m.astype(bool)
    if fill:
        m = ndi.binary_fill_holes(m)
    if largest_cc and m.any():
        n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        if n > 2:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            m = lab == sizes.argmax()
    return m


# --------------------------------------------------------------- pipelines
METHODS = {
    "M1_gray_otsu":   dict(score=score_m1_gray,          morph=False,
                           label="M1: Grayscale + Otsu (baseline)"),
    "M2_colorGMM":    dict(score=score_m2_color,         morph=False,
                           label="M2: Background color GMM + Otsu"),
    "M3_color_center": dict(score=score_m3_color_center, morph=False,
                            label="M3: M2 x center prior + Otsu"),
    "M4_M3_morph":    dict(score=score_m3_color_center,  morph=True,
                           label="M4: M3 + Morphology (open/close/fill/CC)"),
}


def run_method(img_bgr, key):
    """คืน (score map, binary mask, threshold ที่ใช้)"""
    cfg = METHODS[key]
    score = cfg["score"](img_bgr)
    mask, thr = otsu(score)
    if cfg["morph"]:
        mask = postprocess(mask)
    return score, mask, thr
