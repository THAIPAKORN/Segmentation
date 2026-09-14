"""
เตรียมภาพสำหรับทดลอง : หมาพันธุ์เดียว 60 ภาพ ภาพละ 1 ตัว บนพื้นหลัง

ที่มาของภาพ : Oxford-IIIT Pet Dataset (ฟรี ใช้เพื่อการศึกษาได้)
              https://www.robots.ox.ac.uk/~vgg/data/pets/
              ดีตรงที่เขาทำ ground truth ระดับ pixel มาให้แล้ว เราไม่ต้องระบายเอง

ไฟล์ที่ได้
  data/images/  ภาพหมา
  data/truth/   คำตอบที่ถูกต้อง (ขาว = หมา, ดำ = พื้นหลัง)
  data/use/     pixel ที่เอามานับคะแนน (ดำ = ไม่นับ)
"""
import tarfile

import cv2
import numpy as np

from paths import (ANNOTATION_DIR, IMAGES_DIR, IMAGES_TARGZ, TRUTH_DIR, USE_DIR,
                   clear_folder)

BREED = "scottish_terrier"     # หมาพันธุ์เดียว เปลี่ยนชื่อพันธุ์ตรงนี้ได้
HOW_MANY = 60         # โจทย์ขออย่างน้อย 50 ภาพ
MAX_SIDE = 400        # ย่อภาพให้ด้านยาวสุด 400 pixel จะได้รันเร็ว

# ค่าใน trimap ของ Oxford : 1 = ตัวหมา, 2 = พื้นหลัง, 3 = แถบขอบที่กำกวม
DOG, BACKGROUND, UNSURE = 1, 2, 3


def shrink(image, how):
    """ย่อภาพให้ด้านยาวสุดไม่เกิน MAX_SIDE"""
    h, w = image.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=how)


def main():
    for folder in (IMAGES_DIR, TRUTH_DIR, USE_DIR):
        clear_folder(folder)

    # เลือกเฉพาะภาพของพันธุ์ที่ต้องการ ที่มี ground truth ครบ
    names = sorted(p.stem for p in ANNOTATION_DIR.glob(f"{BREED}_*.png"))
    print(f"พันธุ์ {BREED} มีทั้งหมด {len(names)} ภาพ")

    wanted = {f"images/{n}.jpg": n for n in names}
    saved = 0

    with tarfile.open(IMAGES_TARGZ, "r:gz") as tar:
        for item in tar:
            if saved >= HOW_MANY or item.name not in wanted:
                continue
            name = wanted[item.name]

            raw = np.frombuffer(tar.extractfile(item).read(), np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            trimap = cv2.imread(str(ANNOTATION_DIR / f"{name}.png"),
                                cv2.IMREAD_UNCHANGED)
            if image is None or trimap is None:
                continue

            image = shrink(image, cv2.INTER_AREA)
            trimap = shrink(trimap, cv2.INTER_NEAREST)

            truth = (trimap == DOG)
            # ข้ามภาพที่หมาเล็กเกินไปหรือใหญ่เต็มจอ จะได้เหลือแต่ภาพ "หมา 1 ตัวบนพื้นหลัง"
            if not 0.15 < truth.mean() < 0.70:
                continue

            cv2.imwrite(str(IMAGES_DIR / f"{name}.jpg"), image)
            cv2.imwrite(str(TRUTH_DIR / f"{name}.png"), truth.astype(np.uint8) * 255)
            cv2.imwrite(str(USE_DIR / f"{name}.png"),
                        (trimap != UNSURE).astype(np.uint8) * 255)
            saved += 1

    print(f"เก็บไว้ใช้ทดลอง {saved} ภาพ -> {IMAGES_DIR.parent}")


if __name__ == "__main__":
    main()
