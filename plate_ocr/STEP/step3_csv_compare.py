# 3단계 CSV 데이터와 카메라 OCR 결과 비교

# 목표
#   1. OCR은 “입력 데이터”
#   2. CSV는 “기준 데이터”
#   3. 판단은 코드가 한다는 개념 정립

# CSV 예시 (plates.csv)
#   plate
#   152가3018
#   12나3456

step3_csv_compare.py의 완전한 수정본입니다:
python# 3단계 CSV 데이터와 카메라 OCR 결과 비교

# 목표
#   1. OCR은 "입력 데이터"
#   2. CSV는 "기준 데이터"
#   3. 판단은 코드가 한다는 개념 정립

# CSV 예시 (plates.csv)
#   plate
#   152가3018
#   12나3456

import cv2
import pytesseract
import pandas as pd
import re
import os

OCR_KOR = "--oem 3 --psm 6 -l kor+eng"
CSV_PATH = "plates.csv"

# CSV 파일 존재 확인
if not os.path.exists(CSV_PATH):
    print(f"경고: {CSV_PATH} 파일이 없습니다!")
    print("plates.csv 파일을 생성해주세요.")
    print("예시:\nplate\n152가3018\n12나3456")
    exit()

# CSV 로드
df = pd.read_csv(CSV_PATH)
WHITELIST = set(df["plate"].astype(str).str.strip())

def parse_plate(text):
    t = re.sub(r"\s+", "", text)
    m = re.search(r'(\d{2,3})([가-힣])(\d{4})', t)
    if m:
        return m.group(0)
    return None

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

print("ESC 종료")
print(f"등록된 차량 수: {len(WHITELIST)}대")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config=OCR_KOR)

    plate = parse_plate(text)

    if plate:
        if plate in WHITELIST:
            msg = f"허용 차량: {plate}"
            color = (0,255,0)
        else:
            msg = f"미등록 차량: {plate}"
            color = (0,0,255)
    else:
        msg = "번호판 인식 중..."
        color = (255,255,0)

    cv2.putText(frame, msg, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("CSV Compare", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
