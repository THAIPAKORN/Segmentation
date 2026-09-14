"""
รันการทดลองทั้งหมด แล้วสร้างตารางกับรูปสำหรับรายงาน

ทำ 4 อย่าง
  1. รันทุกภาพ ทั้งแบบ "ไม่ใช้ Morphology" และ "ใช้ Morphology" แล้วเทียบกัน
  2. ลองขนาด Structuring Element หลาย ๆ ขนาด ดูว่าขนาดไหนดีที่สุด
  3. วาด ROC curve และคำนวณ AUC
  4. บันทึกรูปทั้งหมดลง results/figures
"""
import csv

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

import segment as S
from paths import (FIGURE_DIR, IMAGES_DIR, TABLE_DIR, TRUTH_DIR, USE_DIR,
                   image_names)

rcParams["font.family"] = ["Leelawadee UI", "Tahoma", "DejaVu Sans"]
rcParams["figure.dpi"] = 130

SE_SIZES = [1, 5, 9, 13, 17, 21, 25, 31, 41]   # ขนาด SE ที่จะลอง (1 = ไม่ทำอะไร)


def load(name):
    """อ่านภาพ 3 ไฟล์ของภาพหนึ่ง ๆ : ภาพจริง, คำตอบ, pixel ที่เอามานับ"""
    image = cv2.imread(str(IMAGES_DIR / f"{name}.jpg"))
    truth = cv2.imread(str(TRUTH_DIR / f"{name}.png"), 0) > 127
    use = cv2.imread(str(USE_DIR / f"{name}.png"), 0) > 127
    return image, truth, use


def save_table(filename, rows):
    with open(TABLE_DIR / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("  บันทึก ->", TABLE_DIR / filename)


# =====================================================================
# 1. รันทุกภาพ
# =====================================================================
def run_everything(names):
    """คืน score map ของทุกภาพ และผลรวม Confusion Matrix ของทั้งสองแบบ"""
    scores, truths, uses = [], [], []
    totals = {"ไม่ใช้ Morphology": [0, 0, 0, 0], "ใช้ Morphology": [0, 0, 0, 0]}
    per_image = []

    for name in names:
        image, truth, use = load(name)
        score = S.make_score(image)
        plain = S.to_black_white(score)
        cleaned = S.clean_with_morphology(plain)

        scores.append(score)
        truths.append(truth)
        uses.append(use)

        for label, mask in (("ไม่ใช้ Morphology", plain), ("ใช้ Morphology", cleaned)):
            counts = S.count_pixels(mask, truth, use)
            totals[label] = [a + b for a, b in zip(totals[label], counts)]
            row = {"ภาพ": name, "วิธี": label}
            row.update({k: round(v, 4) for k, v in S.measures(*counts).items()})
            per_image.append(row)

        # เก็บภาพผลลัพธ์ไว้ดู
        cv2.imwrite(str(FIGURE_DIR.parent / "masks" / f"{name}.png"),
                    cleaned.astype(np.uint8) * 255)

    save_table("ผลรายภาพ.csv", per_image)
    return scores, truths, uses, totals, per_image


# =====================================================================
# 2. ลองขนาด Structuring Element
# =====================================================================
def average_iou(scores, truths, uses, se_size, which):
    """ค่า IoU เฉลี่ยของภาพชุดหนึ่ง เมื่อใช้ SE ขนาดที่กำหนด"""
    values = []
    for i in which:
        mask = S.to_black_white(scores[i])
        if se_size > 1:
            mask = S.clean_with_morphology(mask, se_size)
        tp, fp, fn, _ = S.count_pixels(mask, truths[i], uses[i])
        values.append(tp / (tp + fp + fn + 1e-9))
    return float(np.mean(values))


def try_se_sizes(scores, truths, uses):
    """
    ลองทุกขนาด แล้วเลือกขนาดที่ดีที่สุด

    สำคัญ : เลือกขนาดจาก "ภาพชุดปรับค่า" (ครึ่งแรก) เท่านั้น
            แล้วเอาไปวัดผลจริงกับ "ภาพชุดทดสอบ" (ครึ่งหลัง) ที่ไม่เคยใช้เลือก
            ไม่อย่างนั้นตัวเลขจะดูดีเกินความจริง
    """
    half = len(scores) // 2
    tune, test = range(half), range(half, len(scores))

    rows = []
    for size in SE_SIZES:
        rows.append({
            "ขนาด SE": size,
            "IoU ชุดปรับค่า": round(average_iou(scores, truths, uses, size, tune), 4),
            "IoU ชุดทดสอบ": round(average_iou(scores, truths, uses, size, test), 4),
        })
        print(f"  SE {size:2d}x{size:2d}  ปรับค่า {rows[-1]['IoU ชุดปรับค่า']:.4f}"
              f"   ทดสอบ {rows[-1]['IoU ชุดทดสอบ']:.4f}")

    best = max(rows, key=lambda r: r["IoU ชุดปรับค่า"])
    save_table("ขนาด_SE.csv", rows)
    return rows, best


# =====================================================================
# 3. ROC curve
# =====================================================================
def roc(scores, truths, uses):
    """
    ไล่ threshold ทุกค่าตั้งแต่ 0 ถึง 255 แล้วจดว่าได้ TPR กับ FPR เท่าไร
    (ทายว่าเป็นหมาเมื่อคะแนน > threshold)
    """
    dog = np.zeros(256)
    background = np.zeros(256)
    for score, truth, use in zip(scores, truths, uses):
        dog += np.bincount(score[use & truth], minlength=256)
        background += np.bincount(score[use & ~truth], minlength=256)

    tpr = dog[::-1].cumsum()[::-1] / dog.sum()          # เก็บหมาได้กี่ %
    fpr = background[::-1].cumsum()[::-1] / background.sum()  # เผลอจับพื้นหลังกี่ %

    tpr = np.append(tpr, 0.0)
    fpr = np.append(fpr, 0.0)
    auc = float(np.trapezoid(tpr[::-1], fpr[::-1]))
    return fpr, tpr, auc


# =====================================================================
# 4. รูปทั้งหมด
# =====================================================================
def figure_confusion(totals):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, (label, (tp, fp, fn, tn)) in zip(axes, totals.items()):
        table = np.array([[tp, fn], [fp, tn]], float)
        part = table / table.sum()
        ax.imshow(part, cmap="Blues", vmin=0, vmax=part.max() * 1.15)
        words = [["True Positive (TP)", "False Negative (FN)"],
                 ["False Positive (FP)", "True Negative (TN)"]]
        for i in range(2):
            for j in range(2):
                dark = part[i, j] > part.max() * 0.6
                color = "white" if dark else "black"
                ax.text(j, i - 0.2, words[i][j], ha="center", fontsize=8, color=color)
                ax.text(j, i + 0.1, f"{table[i, j] / 1e6:.2f} M", ha="center",
                        fontsize=13, weight="bold", color=color)
                ax.text(j, i + 0.3, f"{100 * part[i, j]:.1f}%", ha="center",
                        fontsize=8, color=color)
        m = S.measures(tp, fp, fn, tn)
        ax.set_title(f"{label}\nF1={m['f1']:.3f}   IoU={m['iou']:.3f}", fontsize=10)
        ax.set_xticks([0, 1], ["ทายว่าหมา", "ทายว่าพื้นหลัง"], fontsize=9)
        ax.set_yticks([0, 1], ["เป็นหมาจริง", "เป็นพื้นหลังจริง"], fontsize=9)
        ax.set_xticks(np.arange(-.5, 2), minor=True)
        ax.set_yticks(np.arange(-.5, 2), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle("Confusion Matrix (นับ pixel รวมทุกภาพ)", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "รูป1_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def figure_roc(fpr, tpr, auc, totals):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ภาพคะแนนของเรา (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="เดาสุ่ม (AUC=0.500)")
    for (label, (tp, fp, fn, tn)), color in zip(totals.items(), ["#ff7f0e", "#d62728"]):
        ax.plot(fp / (fp + tn), tp / (tp + fn), "o", ms=9, color=color,
                mec="white", label=f"จุดทำงานจริง : {label}")
    ax.set_xlabel("False Positive Rate  (เผลอจับพื้นหลังมาเป็นหมา)")
    ax.set_ylabel("True Positive Rate  (เก็บตัวหมาได้)")
    ax.set_title("ROC curve", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "รูป2_roc.png", bbox_inches="tight")
    plt.close(fig)


def figure_se_sizes(rows, best):
    fig, ax = plt.subplots(figsize=(6.6, 4))
    sizes = [r["ขนาด SE"] for r in rows]
    ax.plot(sizes, [r["IoU ชุดปรับค่า"] for r in rows], "o-", label="ชุดปรับค่า")
    ax.plot(sizes, [r["IoU ชุดทดสอบ"] for r in rows], "s-", label="ชุดทดสอบ")
    ax.axvline(best["ขนาด SE"], color="gray", ls=":")
    ax.annotate(f"เลือกขนาด {best['ขนาด SE']}",
                (best["ขนาด SE"], best["IoU ชุดปรับค่า"]),
                textcoords="offset points", xytext=(8, -12), fontsize=9)
    ax.set_xlabel("ขนาดของ Structuring Element (pixel)   [1 = ไม่ใช้ Morphology]")
    ax.set_ylabel("IoU เฉลี่ย")
    ax.set_title("ขนาด SE มีผลต่อผลลัพธ์อย่างไร", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "รูป3_ขนาด_SE.png", bbox_inches="tight")
    plt.close(fig)


def figure_steps(name):
    """แสดงทีละขั้นให้เห็นว่า Morphology ทำอะไร"""
    image, truth, _ = load(name)
    score = S.make_score(image)
    plain = S.to_black_white(score)
    cleaned = S.clean_with_morphology(plain)

    panels = [(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "1) ภาพต้นฉบับ", None),
              (score, "2) ภาพคะแนน (สว่าง = น่าจะเป็นหมา)", "magma"),
              (plain, "3) ตัดด้วย Otsu", "gray"),
              (cleaned, f"4) เก็บกวาดด้วย Morphology (SE {S.SE_SIZE})", "gray"),
              (truth, "5) คำตอบที่ถูกต้อง", "gray")]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.1))
    for ax, (data, title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"ขั้นตอนการทำงาน ({name})", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "รูป4_ขั้นตอน.png", bbox_inches="tight")
    plt.close(fig)


def figure_examples(per_image, names):
    """ภาพที่ทำได้ดีที่สุด 3 ภาพ และแย่ที่สุด 3 ภาพ"""
    done = sorted([r for r in per_image if r["วิธี"] == "ใช้ Morphology"],
                  key=lambda r: -r["iou"])
    picks = done[:3] + done[-3:]
    fig, axes = plt.subplots(len(picks), 4, figsize=(9, 2.1 * len(picks)))
    for row, item in zip(axes, picks):
        image, truth, _ = load(item["ภาพ"])
        score = S.make_score(image)
        plain = S.to_black_white(score)
        cleaned = S.clean_with_morphology(plain)
        for ax, data, cmap in zip(row,
                                  [cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                   truth, plain, cleaned],
                                  [None, "gray", "gray", "gray"]):
            ax.imshow(data, cmap=cmap)
            ax.axis("off")
        # ชื่อภาพวางไว้มุมซ้ายบนของภาพต้นฉบับ จะได้ไม่ทับหัวคอลัมน์
        row[0].text(0.02, 0.98, item["ภาพ"], fontsize=7, va="top", color="white",
                    transform=row[0].transAxes,
                    bbox=dict(fc="black", ec="none", alpha=.55, pad=1.5))
        row[3].text(0.98, 0.02, f"IoU = {item['iou']:.3f}", fontsize=9, ha="right",
                    color="black", transform=row[3].transAxes,
                    bbox=dict(fc="white", ec="#bbbbbb", pad=2))
    for ax, title in zip(axes[0], ["ภาพต้นฉบับ", "คำตอบที่ถูกต้อง",
                                   "ไม่ใช้ Morphology", "ใช้ Morphology"]):
        ax.set_title(title, fontsize=9.5)
    fig.suptitle("3 แถวบน = ทำได้ดีที่สุด   3 แถวล่าง = แย่ที่สุด",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "รูป5_ตัวอย่าง.png", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
def main():
    for folder in (FIGURE_DIR, TABLE_DIR, FIGURE_DIR.parent / "masks"):
        folder.mkdir(parents=True, exist_ok=True)

    names = image_names()
    print(f"ภาพทั้งหมด {len(names)} ภาพ")

    # แถบขอบที่ ground truth เองก็ไม่แน่ใจ เราตัดไม่นำมานับ ต้องรู้ว่ามันกินไปเท่าไร
    skipped = [1 - load(n)[2].mean() for n in names]
    print(f"ตัดแถบขอบกำกวมออกเฉลี่ย {100 * np.mean(skipped):.1f}% ของ pixel "
          f"(ภาพที่ตัดมากสุด {100 * max(skipped):.1f}%)")

    print("\n[1] รันทุกภาพ")
    scores, truths, uses, totals, per_image = run_everything(names)

    print("\n[2] ลองขนาด Structuring Element")
    se_rows, best = try_se_sizes(scores, truths, uses)
    print(f"  -> ขนาดที่ดีที่สุดคือ {best['ขนาด SE']} "
          f"(ตอนนี้ segment.py ตั้งไว้ที่ {S.SE_SIZE})")

    print("\n[3] ROC curve")
    fpr, tpr, auc = roc(scores, truths, uses)
    print(f"  AUC = {auc:.4f}")

    print("\n[4] สรุปผล")
    summary = []
    for label, counts in totals.items():
        m = S.measures(*counts)
        summary.append({"วิธี": label, "TP": counts[0], "FP": counts[1],
                        "FN": counts[2], "TN": counts[3],
                        **{k: round(v, 4) for k, v in m.items()}, "AUC": round(auc, 4)})
        print(f"  {label:18s} Accuracy={m['accuracy']:.3f}  Precision={m['precision']:.3f}"
              f"  Recall={m['recall']:.3f}  F1={m['f1']:.3f}  IoU={m['iou']:.3f}")
    save_table("สรุปผล.csv", summary)

    print("\n[5] วาดรูป")
    figure_confusion(totals)
    figure_roc(fpr, tpr, auc, totals)
    figure_se_sizes(se_rows, best)
    best_image = max((r for r in per_image if r["วิธี"] == "ใช้ Morphology"),
                     key=lambda r: r["iou"])["ภาพ"]
    figure_steps(best_image)
    figure_examples(per_image, names)
    print("  รูปทั้งหมด ->", FIGURE_DIR)


if __name__ == "__main__":
    main()
