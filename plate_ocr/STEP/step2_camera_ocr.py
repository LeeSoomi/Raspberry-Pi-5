# 2단계 실시간 카메라로 텍스트 검출

# 목표
#   1.  “이미지 1장” → “실시간 프레임” 차이 이해
#   2. 프레임이 계속 바뀌는 데이터라는 개념 습득 
#   3. OCR을 연속으로 하면 왜 불안정한지 체감

# 핵심개념
#   cv2.VideoCapture
#   while 루프
#   실시간 OCR의 한계

import cv2
import pytesseract

OCR_KOR = "--oem 3 --psm 6 -l kor+eng"

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("ESC 누르면 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config=OCR_KOR)

    cv2.putText(frame, "OCR Running...", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    print("[OCR 결과]")
    print(text)

    cv2.imshow("Camera OCR", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
