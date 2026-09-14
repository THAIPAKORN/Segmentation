"""
Workshop ข้อ 3 + 4 : รัน algorithm ทุกวิธีกับทุกภาพ แล้วเก็บผลไว้ประเมิน

ผลลัพธ์ที่บันทึก
  results/predictions/<method>/<name>_score.png : score map 0-255 (ใช้ทำ ROC)
  results/predictions/<method>/<name>_mask.png  : binary mask หลัง Otsu (+morphology)
  results/tables/per_image_metrics.csv          : metric รายภาพ
"""
import csv
import time

import cv2
import numpy as np

import config as C
import methods as M


def confusion(pred, gt, valid):
    """นับ TP/FP/TN/FN เฉพาะ pixel ที่ valid (ไม่ใช่เส้นขอบกำกวม)"""
    p, g = pred[valid], gt[valid]
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(np.count_nonzero(~p & ~g))
    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    e = 1e-12
    prec = tp / (tp + fp + e)
    rec = tp / (tp + fn + e)
    return dict(
        TP=tp, FP=fp, FN=fn, TN=tn,
        accuracy=(tp + tn) / (tp + fp + fn + tn + e),
        precision=prec,
        recall=rec,                       # = TPR = Hit rate
        specificity=tn / (tn + fp + e),   # = TNR
        fpr=fp / (fp + tn + e),           # = Fall-out / False alarm
        fnr=fn / (fn + tp + e),           # = Miss rate
        f1=2 * prec * rec / (prec + rec + e),
        iou=tp / (tp + fp + fn + e),
    )


def main():
    names = [p.stem for p in sorted(C.IMG_DIR.glob("*.jpg"))]
    rows = []
    for key in M.METHODS:
        (C.PRED_DIR / key).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for name in names:
            img = cv2.imread(str(C.IMG_DIR / f"{name}.jpg"))
            gt = cv2.imread(str(C.GT_DIR / f"{name}.png"), 0) > 127
            valid = cv2.imread(str(C.VALID_DIR / f"{name}.png"), 0) > 127

            score, mask, thr = M.run_method(img, key)
            cv2.imwrite(str(C.PRED_DIR / key / f"{name}_score.png"),
                        np.round(score * 255).astype(np.uint8))
            cv2.imwrite(str(C.PRED_DIR / key / f"{name}_mask.png"),
                        mask.astype(np.uint8) * 255)

            r = metrics(*confusion(mask, gt, valid))
            r.update(method=key, name=name, threshold=round(thr, 4))
            rows.append(r)
        n = len(names)
        agg = np.mean([r["iou"] for r in rows[-n:]])
        print(f"{key:18s} {time.time()-t0:6.1f}s  mIoU={agg:.3f}")

    cols = ["method", "name", "threshold", "TP", "FP", "FN", "TN",
            "accuracy", "precision", "recall", "specificity",
            "fpr", "fnr", "f1", "iou"]
    out = C.TAB_DIR / "per_image_metrics.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: (round(r[c], 6) if isinstance(r[c], float) else r[c])
                         for c in cols})
    print("->", out)


if __name__ == "__main__":
    main()
