# 1단계 핵심 개념
# cv2.imread()
# pytesseract.image_to_string()

import cv2
import pytesseract
import os

IMAGE_DIR = "images"
OCR_KOR = "--oem 3 --psm 6 -l kor+eng"

for file in os.listdir(IMAGE_DIR):
    if not file.lower().endswith((".jpg", ".png")):
        continue

    path = os.path.join(IMAGE_DIR, file)
    img = cv2.imread(path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config=OCR_KOR)

    print("=" * 40)
    print(f"[파일명] {file}")
    print("[OCR 결과]")
    print(text)
