"""
Workshop ข้อ 4 : ทดลองใช้ Morphology algorithm เพื่อปรับปรุงผล Segmentation

ตั้งต้นจาก binary mask ของวิธี M3 (ยังไม่ผ่าน morphology) แล้วทดลอง
  (ก) Ablation  : เปิด/ปิดแต่ละขั้น (Opening, Closing, Fill holes, Largest CC)
  (ข) SE grid   : รูปร่าง Structuring Element x ขนาด  (ตาม Lecture 10)
วัดด้วย mean IoU และ mean F1 ต่อภาพ

สำคัญ : แบ่ง 74 ภาพเป็น TUNE 37 / TEST 37 (สายพันธุ์ละ 1 ภาพต่อชุด)
        เลือกค่า morphology จากชุด TUNE เท่านั้น แล้วรายงานผลบนชุด TEST
        ที่ไม่เคยถูกใช้เลือกค่า เพื่อไม่ให้ตัวเลขดูดีเกินจริง (over-tuning)
"""
import csv
import itertools

import cv2
import numpy as np

import config as C
import methods as M

BASE = "M3_color_center"


def load_split():
    """แบ่งภาพเป็นชุด tune / test โดยแต่ละสายพันธุ์ลงคนละชุด"""
    import csv as _csv
    with open(C.META_CSV, encoding="utf-8") as f:
        meta = list(_csv.DictReader(f))

    by_breed = {}
    for r in meta:
        by_breed.setdefault(r["class_id"], []).append(r["name"])

    tune_names, test_names = [], []
    for cid in sorted(by_breed, key=int):
        for i, n in enumerate(sorted(by_breed[cid])):
            (tune_names if i % 2 == 0 else test_names).append(n)

    def load(names):
        return [(
            cv2.imread(str(C.PRED_DIR / BASE / f"{n}_mask.png"), 0) > 127,
            cv2.imread(str(C.GT_DIR / f"{n}.png"), 0) > 127,
            cv2.imread(str(C.VALID_DIR / f"{n}.png"), 0) > 127,
        ) for n in names]

    return load(tune_names), load(test_names), tune_names, test_names


def score(data, **kw):
    """คืน (mean IoU, mean F1) ของทั้ง dataset สำหรับ morphology ชุดหนึ่ง"""
    ious, f1s = [], []
    for mask, gt, valid in data:
        m = M.postprocess(mask, **kw) if kw else mask
        p, g = m[valid], gt[valid]
        tp = np.count_nonzero(p & g)
        fp = np.count_nonzero(p & ~g)
        fn = np.count_nonzero(~p & g)
        ious.append(tp / (tp + fp + fn + 1e-12))
        f1s.append(2 * tp / (2 * tp + fp + fn + 1e-12))
    return float(np.mean(ious)), float(np.mean(f1s))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("->", path)


def main():
    tune, test, tune_names, test_names = load_split()
    print(f"TUNE {len(tune)} / TEST {len(test)} images (split by breed)")
    base_iou, base_f1 = score(tune)
    print(f"ก่อนทำ morphology [TUNE] : mIoU={base_iou:.4f}  mF1={base_f1:.4f}\n")

    # ---------- (ก) ablation ของแต่ละขั้น ----------
    print("=== (ก) Ablation : เปิด/ปิดแต่ละขั้น [TUNE] (SE = ellipse 7x7) ===")
    rows = []
    for o, c, fi, cc in itertools.product([False, True], repeat=4):
        iou, f1 = score(tune, do_open=o, do_close=c, fill=fi, largest_cc=cc)
        rows.append(dict(open=int(o), close=int(c), fill_holes=int(fi),
                         largest_cc=int(cc), mIoU=round(iou, 4),
                         mF1=round(f1, 4), d_mIoU=round(iou - base_iou, 4)))
    rows.sort(key=lambda r: -r["mIoU"])
    for r in rows[:6]:
        print(f"  open={r['open']} close={r['close']} fill={r['fill_holes']} "
              f"cc={r['largest_cc']} | mIoU={r['mIoU']:.4f} "
              f"({r['d_mIoU']:+.4f})  mF1={r['mF1']:.4f}")
    write_csv(C.TAB_DIR / "morph_ablation.csv", rows)
    best_ab = rows[0]

    # ---------- (ข) รูปร่างและขนาดของ Structuring Element ----------
    print("\n=== (ข) Structuring Element : รูปร่าง x ขนาด ===")
    stage = dict(do_open=bool(best_ab["open"]), do_close=bool(best_ab["close"]),
                 fill=bool(best_ab["fill_holes"]),
                 largest_cc=bool(best_ab["largest_cc"]))
    grid = []
    for shape in M.SE_SHAPES:
        line = []
        for size in (3, 5, 7, 9, 11, 15, 21, 27, 35, 45, 61):
            iou, f1 = score(tune, se_shape=shape, se_size=size, **stage)
            grid.append(dict(se_shape=shape, se_size=size,
                             mIoU=round(iou, 4), mF1=round(f1, 4)))
            line.append(f"{size}:{iou:.4f}")
        print(f"  {shape:8s} " + "  ".join(line))
    write_csv(C.TAB_DIR / "morph_se_grid.csv", grid)

    best = max(grid, key=lambda r: r["mIoU"])
    cfg = dict(se_shape=best["se_shape"], se_size=best["se_size"], **stage)
    print()
    print(f"[TUNE] best SE = {best['se_shape']} "
          f"{best['se_size']}x{best['se_size']}   stages = {stage}")

    # ---------- confirm on TEST split (ไม่เคยใช้เลือกค่า) ----------
    print()
    print("=== confirm on held-out TEST split ===")
    rows = []
    for split, data in (("TUNE", tune), ("TEST", test), ("ALL", tune + test)):
        b_iou, b_f1 = score(data)
        a_iou, a_f1 = score(data, **cfg)
        rows.append(dict(split=split, n=len(data),
                         mIoU_before=round(b_iou, 4), mIoU_after=round(a_iou, 4),
                         d_mIoU=round(a_iou - b_iou, 4),
                         mF1_before=round(b_f1, 4), mF1_after=round(a_f1, 4),
                         d_mF1=round(a_f1 - b_f1, 4)))
        print(f"  {split:5s} (n={len(data):2d})  mIoU {b_iou:.4f} -> {a_iou:.4f} "
              f"({a_iou - b_iou:+.4f})   mF1 {b_f1:.4f} -> {a_f1:.4f} "
              f"({a_f1 - b_f1:+.4f})")
    write_csv(C.TAB_DIR / "morph_tune_test.csv", rows)

    with open(C.TAB_DIR / "morph_best_config.txt", "w", encoding="utf-8") as f:
        f.write(repr(cfg) + chr(10))
        f.write("tune=" + ",".join(tune_names) + chr(10))
        f.write("test=" + ",".join(test_names) + chr(10))
    print()
    print("set this in config.py, then re-run 02_run_experiments.py")


if __name__ == "__main__":
    main()
