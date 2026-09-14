# Workshop: แยกหมาออกจากพื้นหลัง (Segmentation + Morphology)

โจทย์จาก **Lecture 10 - Morphological Processing** (หน้า 63)
หมา = Positive class, พื้นหลัง = Negative class

รายงานฉบับเต็ม: https://claude.ai/code/artifact/abbe6c3f-71b7-4b23-bc2c-668702c2048f

## โค้ดมีแค่ 3 ไฟล์

```
src/segment.py          หัวใจของงาน อ่านไฟล์เดียวก็เข้าใจทั้งหมด (~100 บรรทัด)
src/1_prepare_data.py   เตรียมภาพ 60 ภาพ พร้อมคำตอบที่ถูกต้อง
src/2_run_all.py        รันทุกภาพ วัดผล และวาดรูปทั้งหมด
src/paths.py            ที่อยู่ของโฟลเดอร์ต่าง ๆ
```

## วิธีการทำงาน 4 ขั้นตอน

1. หาสีเฉลี่ยของขอบภาพ (10% รอบนอก) = สีพื้นหลัง
2. วัดว่าแต่ละ pixel สีต่างจากพื้นหลังเท่าไร ใน CIE-Lab = "ภาพคะแนน" 0-255
3. ตัดภาพคะแนนเป็นขาว-ดำ ด้วย Otsu
4. เก็บกวาดด้วย Morphology: Closing (SE วงรี 21×21) แล้ว Region filling

## ผลลัพธ์

| | Accuracy | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| ไม่ใช้ Morphology | 0.778 | 0.676 | 0.752 | 0.712 | 0.552 |
| **ใช้ Morphology** | 0.788 | 0.650 | **0.905** | **0.756** | **0.608** |

AUC = 0.814

แบ่งภาพเป็นชุดปรับค่า 30 ภาพ / ชุดทดสอบ 30 ภาพ เลือกขนาด SE จากชุดปรับค่าเท่านั้น
ผลบนชุดทดสอบ IoU เฉลี่ยต่อภาพดีขึ้นจาก 0.583 เป็น 0.620

## ภาพที่ใช้

60 ภาพ หมาพันธุ์ **Scottish Terrier** จาก Oxford-IIIT Pet Dataset
(เขาทำ ground truth ระดับ pixel มาให้แล้ว) เลือกเฉพาะภาพที่ตัวหมากิน 15-70% ของภาพ
เพื่อให้เป็นภาพ "หมา 1 ตัวบนพื้นหลัง" จริง ๆ

- `data/images/` ภาพหมา
- `data/truth/` คำตอบที่ถูกต้อง (ขาว = หมา)
- `data/use/` pixel ที่นำมานับคะแนน (ตัดแถบขอบกำกวมออก เฉลี่ย 13.7% ของภาพ)

**เปลี่ยนพันธุ์หมาได้** โดยแก้ตัวแปร `BREED` บรรทัดเดียวใน `1_prepare_data.py`

## วิธีรัน

ต้องมี `numpy opencv-python matplotlib pillow scipy`

```bash
cd src
python 1_prepare_data.py     # เตรียมภาพ
python 2_run_all.py          # รันทุกอย่าง + สร้างรูป

cd ../report
python build_report.py       # สร้าง index.html
```

ไฟล์ดิบที่ `1_prepare_data.py` ต้องใช้ (ตั้ง path ที่ตัวแปร `RAW` ใน `paths.py`)

```
raw/images.tar.gz        https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
raw/annotations.tar.gz   https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz  (แตกไฟล์แล้ว)
```

## ผลลัพธ์ที่ได้

```
results/tables/สรุปผล.csv      ตัวเลขรวมทั้งสองวิธี
results/tables/ผลรายภาพ.csv    ตัวเลขแยกรายภาพ
results/tables/ขนาด_SE.csv     ผลของขนาด Structuring Element
results/figures/รูป1-5.png     รูปทั้งหมดสำหรับรายงาน
results/masks/                 ผลลัพธ์ขาว-ดำของทุกภาพ
```

## หมายเหตุ

`src/old/` คือโค้ดเวอร์ชันแรกที่ซับซ้อนกว่านี้ (ใช้ GMM, 4 วิธี, 74 ภาพ 37 สายพันธุ์)
เก็บไว้เผื่ออ้างอิง ลบทิ้งได้เลยถ้าไม่ใช้
