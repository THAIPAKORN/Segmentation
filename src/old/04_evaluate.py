"""
Workshop ข้อ 5 : วัดประสิทธิภาพ - Confusion Matrix + ROC Curve

ผลลัพธ์
  results/tables/summary_metrics.csv     : metric รวมทุกวิธี
  results/tables/confusion_matrices.csv  : TP/FP/FN/TN รวมระดับ pixel
  results/tables/roc_points.csv          : จุดบน ROC ของแต่ละวิธี
  results/figures/*.png                  : รูปสำหรับนำเสนอ
หมายเหตุ : นับเฉพาะ pixel ที่ valid (ตัด boundary band ของ trimap ออก)
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
ORDER = ["M1_gray_otsu", "M2_colorGMM", "M3_color_center", "M4_M3_morph"]
SHORT = {"M1_gray_otsu": "M1 Grayscale+Otsu", "M2_colorGMM": "M2 Color GMM",
         "M3_color_center": "M3 +Center prior", "M4_M3_morph": "M4 +Morphology"}
COLORS = {"M1_gray_otsu": "#888888", "M2_colorGMM": "#1f77b4",
          "M3_color_center": "#ff7f0e", "M4_M3_morph": "#d62728"}


def load_gt(name):
    gt = cv2.imread(str(C.GT_DIR / f"{name}.png"), 0) > 127
    valid = cv2.imread(str(C.VALID_DIR / f"{name}.png"), 0) > 127
    return gt, valid


def metrics_from(tp, fp, fn, tn):
    e = 1e-12
    prec, rec = tp / (tp + fp + e), tp / (tp + fn + e)
    return dict(accuracy=(tp + tn) / (tp + fp + fn + tn + e), precision=prec,
                recall=rec, specificity=tn / (tn + fp + e),
                fpr=fp / (fp + tn + e), fnr=fn / (fn + tp + e),
                f1=2 * prec * rec / (prec + rec + e), iou=tp / (tp + fp + fn + e))


# ------------------------------------------------------------ confusion matrix
def total_confusion(method):
    tp = fp = fn = tn = 0
    for n in NAMES:
        pred = cv2.imread(str(C.PRED_DIR / method / f"{n}_mask.png"), 0) > 127
        gt, valid = load_gt(n)
        p, g = pred[valid], gt[valid]
        tp += int(np.count_nonzero(p & g))
        fp += int(np.count_nonzero(p & ~g))
        fn += int(np.count_nonzero(~p & g))
        tn += int(np.count_nonzero(~p & ~g))
    return tp, fp, fn, tn


# ------------------------------------------------------------ ROC จาก score map
def score_histograms(method):
    """สร้าง histogram 256 ช่องของ score แยกตาม class -> ใช้คำนวณ ROC ได้ครบทุก pixel"""
    pos = np.zeros(256, np.int64)
    neg = np.zeros(256, np.int64)
    for n in NAMES:
        s = cv2.imread(str(C.PRED_DIR / method / f"{n}_score.png"), 0)
        gt, valid = load_gt(n)
        pos += np.bincount(s[valid & gt].ravel(), minlength=256)
        neg += np.bincount(s[valid & ~gt].ravel(), minlength=256)
    return pos, neg


def roc_from_hist(pos, neg):
    """threshold ที่ t : ทำนายเป็น Positive เมื่อ score > t"""
    tp = pos[::-1].cumsum()[::-1]
    fp = neg[::-1].cumsum()[::-1]
    tpr = np.r_[1.0, tp / pos.sum()]
    fpr = np.r_[1.0, fp / neg.sum()]
    tpr = np.r_[tpr, 0.0]
    fpr = np.r_[fpr, 0.0]
    o = np.argsort(fpr)
    return fpr[o], tpr[o], float(np.trapezoid(tpr[o], fpr[o]))


def sweep_roc(method, morph):
    """กวาด threshold แล้วใส่ morphology ทุกจุด -> ROC ของ 'ทั้ง pipeline'"""
    ts = np.r_[0, np.arange(5, 251, 6), 255]
    acc = np.zeros((len(ts), 4), np.int64)
    cache = [(cv2.imread(str(C.PRED_DIR / method / f"{n}_score.png"), 0),
              *load_gt(n)) for n in NAMES]
    for i, t in enumerate(ts):
        for s, gt, valid in cache:
            m = s > t
            if morph:
                m = M.postprocess(m)
            p, g = m[valid], gt[valid]
            acc[i] += (np.count_nonzero(p & g), np.count_nonzero(p & ~g),
                       np.count_nonzero(~p & g), np.count_nonzero(~p & ~g))
    tp, fp, fn, tn = acc.T
    tpr = np.r_[1.0, tp / (tp + fn), 0.0]
    fpr = np.r_[1.0, fp / (fp + tn), 0.0]
    o = np.argsort(fpr)
    return fpr[o], tpr[o], float(np.trapezoid(tpr[o], fpr[o])), ts


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("->", path)


# ------------------------------------------------------------ figures
def draw_confusion(ax, tp, fp, fn, tn, title):
    """วาดตารางแบบเดียวกับสไลด์ : แถว = Actual, คอลัมน์ = Predicted"""
    cm = np.array([[tp, fn], [fp, tn]], float)
    frac = cm / cm.sum()
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=frac.max() * 1.15)
    labels = [["True Positive (TP)", "False Negative (FN)"],
              ["False Positive (FP)", "True Negative (TN)"]]
    for i in range(2):
        for j in range(2):
            dark = frac[i, j] > frac.max() * 0.6
            ax.text(j, i - 0.22, labels[i][j], ha="center", va="center",
                    fontsize=7.5, color="white" if dark else "#333")
            ax.text(j, i + 0.08, f"{cm[i, j] / 1e6:,.2f} M", ha="center",
                    va="center", fontsize=12, weight="bold",
                    color="white" if dark else "black")
            ax.text(j, i + 0.28, f"{100 * frac[i, j]:.1f}% ของ pixel ทั้งหมด",
                    ha="center", va="center", fontsize=7,
                    color="white" if dark else "#444")
    ax.set_xticks([0, 1], ["Predicted Positive", "Predicted Negative"], fontsize=8)
    ax.set_yticks([0, 1], ["Actual Positive (สัตว์)", "Actual Negative (พื้นหลัง)"],
                  fontsize=8)
    m = metrics_from(tp, fp, fn, tn)
    ax.set_title(f"{title}    Acc={m['accuracy']:.3f}  P={m['precision']:.3f}  "
                 f"R={m['recall']:.3f}  F1={m['f1']:.3f}", fontsize=9)
    ax.set_xticks(np.arange(-.5, 2), minor=True)
    ax.set_yticks(np.arange(-.5, 2), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)


def fig_confusion(conf):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    draw_confusion(axes[0], *conf["M1_gray_otsu"], "M1 Baseline")
    draw_confusion(axes[1], *conf["M4_M3_morph"], "M4 วิธีที่เสนอ + Morphology")
    fig.suptitle("Confusion Matrix ระดับ pixel รวมทั้ง dataset (74 ภาพ)",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig1_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def fig_roc_scores(rocs, op_points):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for k in ["M1_gray_otsu", "M2_colorGMM", "M3_color_center"]:
        fpr, tpr, auc = rocs[k]
        ax.plot(fpr, tpr, color=COLORS[k], lw=1.9, label=f"{SHORT[k]}  (AUC={auc:.3f})")
        f, t = op_points[k]
        ax.plot(f, t, "o", color=COLORS[k], ms=6, mec="white", mew=1.2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC=0.500)")
    ax.set_xlabel("False Positive Rate  (FP / N)  = Fall-out")
    ax.set_ylabel("True Positive Rate  (TP / P)  = Recall / Hit rate")
    ax.set_title("ROC curve ของ score map แต่ละวิธี" +
                 "        (จุดวงกลม = จุดทำงานจริงเมื่อ threshold ด้วย Otsu)",
                 fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig2_roc_score_maps.png", bbox_inches="tight")
    plt.close(fig)


def fig_roc_morph(no_morph, with_morph, op_points):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for (fpr, tpr, auc, _), c, lab in (
            (no_morph, "#ff7f0e", "M3 ก่อนทำ morphology"),
            (with_morph, "#d62728", "M4 หลังทำ morphology (close 21 + fill)")):
        ax.plot(fpr, tpr, color=c, lw=2, marker="o", ms=2.5,
                label=f"{lab}  (AUC={auc:.3f})")
    for k, c in (("M3_color_center", "#ff7f0e"), ("M4_M3_morph", "#d62728")):
        f, t = op_points[k]
        ax.plot(f, t, "*", color=c, ms=15, mec="white", mew=1)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ผลของ Morphology ต่อ ROC (กวาด threshold แล้วใส่ morphology ทุกจุด)"
                 + "      ดาว = จุดทำงานจริงด้วย Otsu", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig3_roc_morphology_effect.png", bbox_inches="tight")
    plt.close(fig)


def fig_bars(summary):
    keys = ["accuracy", "precision", "recall", "specificity", "f1", "iou"]
    labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "IoU"]
    x = np.arange(len(keys))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.6, 4))
    for i, k in enumerate(ORDER):
        v = [summary[k][m] for m in keys]
        b = ax.bar(x + (i - 1.5) * w, v, w, label=SHORT[k], color=COLORS[k])
        ax.bar_label(b, fmt="%.3f", fontsize=5.6, padding=1)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ค่า")
    ax.set_title("เปรียบเทียบ metric ทุกวิธี (คิดจาก pixel รวมทั้ง dataset)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig4_metric_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _overlay(img, mask, color):
    out = img.copy()
    out[mask] = (0.45 * out[mask] + 0.55 * np.array(color)).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def fig_qualitative(per_image):
    """ตัวอย่างที่ทำได้ดีที่สุด 3 ภาพ และแย่ที่สุด 3 ภาพ (ตัดสินด้วย IoU ของ M4)"""
    m4 = sorted([r for r in per_image if r["method"] == "M4_M3_morph"],
                key=lambda r: -float(r["iou"]))
    picks = m4[:3] + m4[-3:]
    cols = ["ภาพต้นฉบับ", "Ground truth", "M1 baseline", "M3 ก่อน morphology",
            "M4 หลัง morphology"]
    fig, axes = plt.subplots(len(picks), 5, figsize=(11, 2.05 * len(picks)))
    for r, row in zip(picks, axes):
        n = r["name"]
        img = cv2.imread(str(C.IMG_DIR / f"{n}.jpg"))
        gt, _ = load_gt(n)
        row[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        row[1].imshow(_overlay(img, gt, (0, 200, 0)))
        for ax, k in zip(row[2:], ["M1_gray_otsu", "M3_color_center", "M4_M3_morph"]):
            pred = cv2.imread(str(C.PRED_DIR / k / f"{n}_mask.png"), 0) > 127
            iou = next(float(x["iou"]) for x in per_image
                       if x["method"] == k and x["name"] == n)
            ax.imshow(_overlay(img, pred, (0, 0, 230)))
            ax.set_xlabel(f"IoU={iou:.3f}", fontsize=7.5, labelpad=1)
        row[0].set_ylabel(n, fontsize=6.5)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, t in zip(axes[0], cols):
        ax.set_title(t, fontsize=8.5)
    fig.suptitle("ตัวอย่างผลลัพธ์ : 3 แถวบน = ดีที่สุด, 3 แถวล่าง = แย่ที่สุด "
                 "(วัดด้วย IoU ของ M4)", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig5_qualitative_examples.png", bbox_inches="tight")
    plt.close(fig)


def fig_pipeline(name):
    """แสดงทีละขั้นของ pipeline เพื่ออธิบายว่า morphology ทำอะไร"""
    img = cv2.imread(str(C.IMG_DIR / f"{name}.jpg"))
    gt, _ = load_gt(name)
    score = cv2.imread(str(C.PRED_DIR / "M3_color_center" / f"{name}_score.png"), 0)
    raw = cv2.imread(str(C.PRED_DIR / "M3_color_center" / f"{name}_mask.png"), 0) > 127
    se = M.structuring_element()
    closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, se).astype(bool)
    final = M.postprocess(raw)

    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "1) ภาพต้นฉบับ", None),
        (score, "2) Score map (ยิ่งสว่าง = ยิ่งน่าจะเป็นวัตถุ)", "magma"),
        (raw, "3) Threshold ด้วย Otsu", "gray"),
        (closed, f"4) Closing ด้วย SE {C.SE_SHAPE} {C.SE_SIZE}x{C.SE_SIZE}", "gray"),
        (final, "5) + Region filling = ผลสุดท้าย", "gray"),
        (_overlay(img, final, (0, 0, 230)), "6) ทาบกับภาพจริง (เขียว = ขอบ GT)", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4))
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    cs = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in cs:
        axes[1, 2].plot(c[:, 0, 0], c[:, 0, 1], color="#00ff00", lw=1.2)
    fig.suptitle(f"ขั้นตอนการทำงานของ pipeline ({name})", fontsize=10.5, weight="bold")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig6_pipeline_steps.png", bbox_inches="tight")
    plt.close(fig)


def fig_se_study():
    with open(C.TAB_DIR / "morph_se_grid.csv", encoding="utf-8") as f:
        grid = list(csv.DictReader(f))
    with open(C.TAB_DIR / "morph_ablation.csv", encoding="utf-8") as f:
        abl = list(csv.DictReader(f))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.1))
    for shape, c in (("rect", "#1f77b4"), ("ellipse", "#d62728"), ("cross", "#2ca02c")):
        rows = [r for r in grid if r["se_shape"] == shape]
        xs = [int(r["se_size"]) for r in rows]
        ys = [float(r["mIoU"]) for r in rows]
        a1.plot(xs, ys, "o-", color=c, ms=4, label=f"SE = {shape}")
    best = max(grid, key=lambda r: float(r["mIoU"]))
    a1.axvline(int(best["se_size"]), color="gray", ls=":", lw=1)
    a1.annotate("เลือกใช้ " + best["se_shape"] + " " + best["se_size"]
                + "x" + best["se_size"],
                (int(best["se_size"]), float(best["mIoU"])),
                textcoords="offset points", xytext=(6, -14), fontsize=8)
    a1.set_xlabel("ขนาด Structuring Element (pixel)")
    a1.set_ylabel("mean IoU (ชุด TUNE)")
    a1.set_title("(ก) ผลของรูปร่างและขนาด SE", fontsize=9.5)
    a1.legend(fontsize=8)
    a1.grid(alpha=.3)

    abl = sorted(abl, key=lambda r: float(r["mIoU"]))[-8:]
    names = [("Open " if int(r["open"]) else "") + ("Close " if int(r["close"]) else "")
             + ("Fill " if int(r["fill_holes"]) else "")
             + ("LargestCC" if int(r["largest_cc"]) else "") or "ไม่ทำอะไร"
             for r in abl]
    vals = [float(r["mIoU"]) for r in abl]
    bars = a2.barh(names, vals, color="#4c78a8")
    a2.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
    a2.set_xlim(0.5, max(vals) * 1.06)
    a2.set_xlabel("mean IoU (ชุด TUNE)")
    a2.set_title("(ข) Ablation : เปิด/ปิดแต่ละขั้น (SE ellipse 7x7)", fontsize=9.5)
    a2.tick_params(labelsize=7.5)
    a2.grid(axis="x", alpha=.3)
    fig.suptitle("โจทย์ข้อ 4 : การเลือกพารามิเตอร์ของ Morphology",
                 fontsize=10.5, weight="bold")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig7_morphology_study.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ main
def main():
    with open(C.TAB_DIR / "per_image_metrics.csv", encoding="utf-8") as f:
        per_image = list(csv.DictReader(f))

    conf, summary, roc_rows, op_points, rocs = {}, {}, [], {}, {}
    print("=== Confusion matrix ระดับ pixel (รวมทั้ง 74 ภาพ) ===")
    for k in ORDER:
        conf[k] = total_confusion(k)
        summary[k] = metrics_from(*conf[k])
        tp, fp, fn, tn = conf[k]
        op_points[k] = (fp / (fp + tn), tp / (tp + fn))
        print(f"{SHORT[k]:20s} TP={tp:>10,} FP={fp:>10,} FN={fn:>10,} TN={tn:>10,}")

    for k in ORDER[:3]:
        pos, neg = score_histograms(k)
        rocs[k] = roc_from_hist(pos, neg)
        for f_, t_ in zip(*rocs[k][:2]):
            roc_rows.append(dict(method=k, kind="score", fpr=round(f_, 6),
                                 tpr=round(t_, 6)))

    print()
    print("=== ROC ===")
    for k in ORDER[:3]:
        print(f"{SHORT[k]:20s} AUC={rocs[k][2]:.4f}")
    nm = sweep_roc("M3_color_center", morph=False)
    wm = sweep_roc("M3_color_center", morph=True)
    print(f"sweep M3 (no morph)   AUC={nm[2]:.4f}")
    print(f"sweep M4 (with morph) AUC={wm[2]:.4f}")
    for tag, r in (("sweep_no_morph", nm), ("sweep_with_morph", wm)):
        for f_, t_ in zip(r[0], r[1]):
            roc_rows.append(dict(method=tag, kind="sweep", fpr=round(f_, 6),
                                 tpr=round(t_, 6)))

    rows = []
    for k in ORDER:
        tp, fp, fn, tn = conf[k]
        r = dict(method=k, label=SHORT[k], TP=tp, FP=fp, FN=fn, TN=tn)
        r.update({m: round(v, 4) for m, v in summary[k].items()})
        r["AUC"] = round(rocs[k][2], 4) if k in rocs else round(wm[2], 4)
        per = [float(x["iou"]) for x in per_image if x["method"] == k]
        r["mIoU_per_image"] = round(float(np.mean(per)), 4)
        rows.append(r)
    write_csv(C.TAB_DIR / "summary_metrics.csv", rows)
    write_csv(C.TAB_DIR / "confusion_matrices.csv",
              [dict(method=k, TP=conf[k][0], FP=conf[k][1],
                    FN=conf[k][2], TN=conf[k][3]) for k in ORDER])
    write_csv(C.TAB_DIR / "roc_points.csv", roc_rows)

    fig_confusion(conf)
    fig_roc_scores(rocs, op_points)
    fig_roc_morph(nm, wm, op_points)
    fig_bars(summary)
    fig_qualitative(per_image)
    fig_se_study()
    best = max((r for r in per_image if r["method"] == "M4_M3_morph"),
               key=lambda r: float(r["iou"]))
    fig_pipeline(best["name"])
    print()
    print("figures ->", C.FIG_DIR)


if __name__ == "__main__":
    main()
