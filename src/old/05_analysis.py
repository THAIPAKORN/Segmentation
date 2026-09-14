"""
Workshop ข้อ 5 (ส่วนอภิปรายผล) : morphology ช่วยเพราะอะไรกันแน่ ?

จาก fig3 พบว่า ROC ก่อน/หลัง morphology แทบทับกัน (AUC 0.858 vs 0.859)
แต่จุดทำงานจริงกลับดีขึ้นมาก จึงต้องตรวจสอบว่า
  สมมติฐาน A : morphology แค่ทำให้ mask ขยายตัว = เทียบเท่ากับการลด threshold
  สมมติฐาน B : morphology ให้ประโยชน์จริงที่ threshold ใด ๆ ก็ตาม

วิธีตรวจ : กวาด threshold ทั้งช่วง แล้วเทียบ "ค่าดีที่สุดที่ทำได้" ของทั้งสองแบบ
           ถ้าค่าดีที่สุดเท่ากัน แปลว่า A ถูก

นอกจากนี้ยังวัด
  - ความทนต่อการตั้ง threshold (ช่วง threshold ที่ยังให้ผลดี)
  - ความต่อเนื่องเชิงพื้นที่ของผลลัพธ์ (จำนวน connected component, ความยาวเส้นขอบ)
"""
import csv

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

import config as C
import methods as M

rcParams["font.family"] = ["Leelawadee UI", "Tahoma", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130

NAMES = [p.stem for p in sorted(C.IMG_DIR.glob("*.jpg"))]
THRESHOLDS = np.arange(5, 251, 5)


def load_cache():
    return [(cv2.imread(str(C.PRED_DIR / "M3_color_center" / f"{n}_score.png"), 0),
             cv2.imread(str(C.GT_DIR / f"{n}.png"), 0) > 127,
             cv2.imread(str(C.VALID_DIR / f"{n}.png"), 0) > 127) for n in NAMES]


def sweep(cache, morph):
    """คืน (mIoU, mF1) ต่อ threshold"""
    ious, f1s = [], []
    for t in THRESHOLDS:
        a, b = [], []
        for s, gt, v in cache:
            m = M.postprocess(s > t) if morph else (s > t)
            p, g = m[v], gt[v]
            tp = np.count_nonzero(p & g)
            fp = np.count_nonzero(p & ~g)
            fn = np.count_nonzero(~p & g)
            a.append(tp / (tp + fp + fn + 1e-12))
            b.append(2 * tp / (2 * tp + fp + fn + 1e-12))
        ious.append(float(np.mean(a)))
        f1s.append(float(np.mean(b)))
    return np.array(ious), np.array(f1s)


def spatial_stats():
    """จำนวนก้อน (connected component) และความยาวเส้นขอบต่อพื้นที่ ก่อน/หลัง morphology"""
    out = {"before": [], "after": []}
    for n in NAMES:
        raw = cv2.imread(str(C.PRED_DIR / "M3_color_center" / f"{n}_mask.png"), 0) > 127
        fin = cv2.imread(str(C.PRED_DIR / "M4_M3_morph" / f"{n}_mask.png"), 0) > 127
        for key, m in (("before", raw), ("after", fin)):
            u8 = m.astype(np.uint8)
            ncc = cv2.connectedComponents(u8, connectivity=8)[0] - 1
            per = sum(cv2.arcLength(c, True) for c in
                      cv2.findContours(u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0])
            area = max(int(m.sum()), 1)
            out[key].append((ncc, per / np.sqrt(area)))
    return out


def main():
    cache = load_cache()
    iou_no, f1_no = sweep(cache, morph=False)
    iou_mo, f1_mo = sweep(cache, morph=True)

    i_no, i_mo = int(iou_no.argmax()), int(iou_mo.argmax())
    otsu_t = np.mean([float(r["threshold"]) for r in
                      csv.DictReader(open(C.TAB_DIR / "per_image_metrics.csv",
                                          encoding="utf-8"))
                      if r["method"] == "M3_color_center"]) * 255

    print("=== morphology ช่วยจริงแค่ไหน เมื่อเทียบกับการปรับ threshold เฉย ๆ ===")
    print(f"ไม่ใช้ morphology : threshold ดีที่สุด t={THRESHOLDS[i_no]:3d} "
          f"-> mIoU={iou_no[i_no]:.4f}  mF1={f1_no[i_no]:.4f}")
    print(f"ใช้ morphology   : threshold ดีที่สุด t={THRESHOLDS[i_mo]:3d} "
          f"-> mIoU={iou_mo[i_mo]:.4f}  mF1={f1_mo[i_mo]:.4f}")
    print(f"ส่วนต่างเมื่อทั้งคู่ปรับ threshold ดีที่สุดแล้ว : "
          f"{iou_mo[i_mo] - iou_no[i_no]:+.4f} mIoU")
    print(f"(threshold ที่ Otsu เลือกจริงโดยเฉลี่ย t = {otsu_t:.0f})")

    # ความทนต่อการตั้ง threshold : ช่วงที่ยังทำได้ >= 95% ของค่าดีที่สุดของตัวเอง
    def width(v):
        ok = THRESHOLDS[v >= 0.95 * v.max()]
        return int(ok.min()), int(ok.max()), int(ok.max() - ok.min())

    w_no, w_mo = width(iou_no), width(iou_mo)
    print()
    print("=== ความทนต่อการตั้ง threshold (ช่วงที่ยังได้ >= 95% ของค่าดีที่สุด) ===")
    print(f"ไม่ใช้ morphology : t = {w_no[0]}-{w_no[1]}  (กว้าง {w_no[2]})")
    print(f"ใช้ morphology   : t = {w_mo[0]}-{w_mo[1]}  (กว้าง {w_mo[2]})  "
          f"กว้างขึ้น {w_mo[2] / max(w_no[2], 1):.1f} เท่า")

    st = spatial_stats()
    ncc_b = np.mean([x[0] for x in st["before"]])
    ncc_a = np.mean([x[0] for x in st["after"]])
    per_b = np.mean([x[1] for x in st["before"]])
    per_a = np.mean([x[1] for x in st["after"]])
    print()
    print("=== ความต่อเนื่องเชิงพื้นที่ของผลลัพธ์ ===")
    print(f"จำนวนก้อนเฉลี่ยต่อภาพ      : {ncc_b:.1f} -> {ncc_a:.1f}")
    print(f"ความยาวเส้นขอบ/sqrt(พื้นที่) : {per_b:.2f} -> {per_a:.2f}  "
          f"(ยิ่งน้อย = ขอบยิ่งเรียบ)")

    with open(C.TAB_DIR / "threshold_sweep.csv", "w", newline="",
              encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["threshold", "mIoU_no_morph", "mF1_no_morph",
                     "mIoU_with_morph", "mF1_with_morph"])
        for i, t in enumerate(THRESHOLDS):
            wr.writerow([int(t), round(iou_no[i], 4), round(f1_no[i], 4),
                         round(iou_mo[i], 4), round(f1_mo[i], 4)])
    with open(C.TAB_DIR / "morph_vs_threshold.csv", "w", newline="",
              encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["setting", "best_threshold", "best_mIoU", "best_mF1",
                     "robust_range", "mean_components", "perimeter_ratio"])
        wr.writerow(["no_morphology", int(THRESHOLDS[i_no]), round(iou_no[i_no], 4),
                     round(f1_no[i_no], 4), f"{w_no[0]}-{w_no[1]}",
                     round(ncc_b, 2), round(per_b, 2)])
        wr.writerow(["with_morphology", int(THRESHOLDS[i_mo]), round(iou_mo[i_mo], 4),
                     round(f1_mo[i_mo], 4), f"{w_mo[0]}-{w_mo[1]}",
                     round(ncc_a, 2), round(per_a, 2)])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(THRESHOLDS, iou_no, "-o", ms=3, color="#ff7f0e",
            label="ไม่ใช้ morphology")
    ax.plot(THRESHOLDS, iou_mo, "-o", ms=3, color="#d62728", label="ใช้ morphology")
    ax.axvline(otsu_t, color="gray", ls="--", lw=1)
    ax.annotate("threshold ที่ Otsu เลือก", (otsu_t, 0.06),
                textcoords="offset points", xytext=(5, 0), fontsize=8, color="gray")
    ax.plot(THRESHOLDS[i_no], iou_no[i_no], "*", ms=15, color="#ff7f0e",
            mec="white", mew=1)
    ax.plot(THRESHOLDS[i_mo], iou_mo[i_mo], "*", ms=15, color="#d62728",
            mec="white", mew=1)
    ax.axhspan(0, 0, color="none")
    ax.set_xlabel("threshold ที่ใช้ตัด score map (0-255)")
    ax.set_ylabel("mean IoU")
    ax.set_title("morphology ไม่ได้ยกเพดานความแม่นยำมากนัก "
                 "แต่ทำให้ทนต่อการตั้ง threshold ได้กว้างขึ้นมาก", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    ax.set_ylim(0, max(iou_mo.max(), iou_no.max()) * 1.15)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig8_threshold_robustness.png", bbox_inches="tight")
    plt.close(fig)
    print()
    print("->", C.FIG_DIR / "fig8_threshold_robustness.png")


if __name__ == "__main__":
    main()
