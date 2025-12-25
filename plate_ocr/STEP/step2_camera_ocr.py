# 2단계 실시간 카메라로 텍스트 검출

# 목표
#   1.  “이미지 1장” → “실시간 프레임” 차이 이해
#   2. 프레임이 계속 바뀌는 데이터라는 개념 습득 
#   3. OCR을 연속으로 하면 왜 불안정한지 체감

# 핵심개념
#   cv2.VideoCapture
#   while 루프
#   실시간 OCR의 한계

# 2단계 실시간 카메라로 텍스트 검출

import cv2
import pytesseract

OCR_KOR = "--oem 3 --psm 6 -l kor+eng"

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("ESC 누르면 종료")

# 프레임 스킵으로 부하 감소
frame_count = 0
OCR_INTERVAL = 30  # 30프레임마다 1번 OCR
last_text = ""  # 마지막 OCR 결과 저장

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_count += 1
    
    # OCR 수행
    if frame_count % OCR_INTERVAL == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config=OCR_KOR)
        
        # 텍스트가 있을 때만 출력
        if text.strip():
            last_text = text.strip()
            print("[OCR 결과]")
            print(last_text)
            print("-" * 40)
    
    # 상태 표시
    cv2.putText(frame, f"OCR Running... (Frame: {frame_count})", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    # 마지막 인식 결과 표시
    if last_text:
        cv2.putText(frame, "Last: " + last_text[:20], (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Camera OCR", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
