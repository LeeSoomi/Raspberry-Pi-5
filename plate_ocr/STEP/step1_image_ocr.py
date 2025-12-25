# 1단계 같은 폴더에 있는 이미지에서 텍스트 검출

# 목표
# 1. OCR이 “카메라가 없어도” 동작함을 이해
# 2. 이미지 → 텍스트 변환의 기본 구조 이해
# 3. 실패해도 부담 없는 단계

# 핵심 개념
#     cv2.imread()
#     pytesseract.image_to_string()
#     OCR 결과는 완벽하지 않다

# 폴더구조
# project/
#  ├─ images/
#  │   ├─ car1.jpg
#  │   ├─ car2.jpg
#  ├─ step1_image_ocr.py


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
    
    cv2.imshow("Original", img)
    cv2.imshow("Grayscale", gray)
    
    text = pytesseract.image_to_string(gray, config=OCR_KOR)

    print("=" * 40)
    print(f"[파일명] {file}")
    print("[OCR 결과]")
    print(text)
    print("\n아무 키나 누르면 다음 이미지로 넘어갑니다...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
