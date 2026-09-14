"""
Workshop ข้อ 2 : สร้าง Dataset >= 50 ภาพ พร้อม Ground Truth

ที่มา : Oxford-IIIT Pet Dataset (Parkhi et al., CVPR 2012)
        https://www.robots.ox.ac.uk/~vgg/data/pets/
        7,349 ภาพ สัตว์เลี้ยง 37 สายพันธุ์ พร้อม trimap ระดับ pixel

การคัดเลือก : สุ่มแบบ stratified สายพันธุ์ละ 2 ภาพ -> 74 ภาพ
              (ครอบคลุมทั้ง 12 สายพันธุ์แมว และ 25 สายพันธุ์สุนัข)

Ground truth : trimap 1=ตัวสัตว์, 2=พื้นหลัง, 3=เส้นขอบกำกวม
               -> GT binary  : 255 เมื่อ trimap==1 (Positive = ตัวสัตว์)
               -> valid mask : 0 เมื่อ trimap==3 (ตัดออกจากการวัดผล)
"""
import csv
import random
import tarfile

import cv2
import numpy as np

import config as C


def read_list():
    """อ่าน list.txt -> [(name, class_id, species, breed_id), ...]"""
    rows = []
    with open(C.LIST_TXT) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, cid, sp, bid = line.split()
            rows.append((name, int(cid), int(sp), int(bid)))
    return rows


def stratified_sample(rows):
    """สุ่มสายพันธุ์ละ C.PER_BREED ภาพ โดยรับเฉพาะภาพที่มี trimap จริง"""
    by_breed = {}
    for name, cid, sp, bid in rows:
        if not (C.TRIMAP_DIR / f"{name}.png").exists():
            continue
        by_breed.setdefault(cid, []).append((name, cid, sp, bid))

    rng = random.Random(C.SEED)
    picked = []
    for cid in sorted(by_breed):
        pool = sorted(by_breed[cid])
        picked += rng.sample(pool, min(C.PER_BREED, len(pool)))
    return picked


def resize_keep_ratio(img, max_side, interp):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)


def main():
    picked = stratified_sample(read_list())
    wanted = {f"images/{n}.jpg": (n, cid, sp, bid) for n, cid, sp, bid in picked}
    print(f"เลือกไว้ {len(wanted)} ภาพ จาก {len({p[1] for p in picked})} สายพันธุ์")

    meta, found = [], 0
    with tarfile.open(C.RAW_IMAGES_TGZ, "r:gz") as tar:
        for member in tar:                      # อ่านแบบ sequential (เร็วกว่าสุ่มหา)
            if member.name not in wanted:
                continue
            name, cid, sp, bid = wanted[member.name]

            buf = np.frombuffer(tar.extractfile(member).read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            trimap = cv2.imread(str(C.TRIMAP_DIR / f"{name}.png"), cv2.IMREAD_UNCHANGED)
            if img is None or trimap is None:
                print("  ! ข้าม", name)
                continue
            if img.shape[:2] != trimap.shape[:2]:
                trimap = cv2.resize(trimap, (img.shape[1], img.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

            img = resize_keep_ratio(img, C.MAX_SIDE, cv2.INTER_AREA)
            trimap = resize_keep_ratio(trimap, C.MAX_SIDE, cv2.INTER_NEAREST)

            gt = np.where(trimap == C.TRIMAP_FG, 255, 0).astype(np.uint8)
            valid = np.where(trimap == C.TRIMAP_BOUNDARY, 0, 255).astype(np.uint8)

            cv2.imwrite(str(C.IMG_DIR / f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(C.GT_DIR / f"{name}.png"), gt)
            cv2.imwrite(str(C.VALID_DIR / f"{name}.png"), valid)

            h, w = gt.shape
            n_valid = int((valid > 0).sum())
            meta.append(dict(
                name=name, species="cat" if sp == 1 else "dog", class_id=cid,
                breed_id=bid, height=h, width=w,
                fg_ratio=round(float((gt > 0).sum()) / n_valid, 4),
                valid_ratio=round(n_valid / (h * w), 4),
            ))
            found += 1

    meta.sort(key=lambda r: r["name"])
    with open(C.META_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        wr.writeheader()
        wr.writerows(meta)

    fg = np.array([r["fg_ratio"] for r in meta])
    n_cat = sum(r["species"] == "cat" for r in meta)
    print(f"\nสร้าง dataset สำเร็จ {found} ภาพ  (แมว {n_cat} / สุนัข {found - n_cat})")
    print(f"สัดส่วน Positive (ตัวสัตว์) เฉลี่ย {fg.mean():.3f}  "
          f"[min {fg.min():.3f}, max {fg.max():.3f}]")
    print(f"metadata -> {C.META_CSV}")


if __name__ == "__main__":
    main()
