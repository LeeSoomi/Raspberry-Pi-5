#!/usr/bin/env python3
"""
Raspberry Pi 5
Ultrasonic → OCR → CSV Compare → LED
"""

import cv2
import pytesseract
import time
import re
import pandas as pd
from gpiozero import DistanceSensor, LED

# ======================
# GPIO / HARDWARE
# ======================
ultrasonic = DistanceSensor(echo=27, trigger=17, max_distance=2.0)

LED_RED = LED(23)
LED_GREEN = LED(12)
LED_YELLOW = LED(20)

def set_led(red=False, green=False, yellow=False):
    LED_RED.on() if red else LED_RED.off()
    LED_GREEN.on() if green else LED_GREEN.off()
    LED_YELLOW.on() if yellow else LED_YELLOW.off()

# ======================
# OCR / CONFIG
# ======================
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

OCR_KOR = "--oem 3 --psm 6 -l kor+eng"
OCR_NUM = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

OCR_INTERVAL = 1.5        # OCR 실행 최소 간격 (초)
RESULT_HOLD_SEC = 3.0     # 결과 LED 유지 시간
DIST_THRESHOLD_CM = 40.0  # 초음파 트리거 거리

last_ocr_time = 0
last_result_time = 0
state = "SCAN"
last_digits = None
last_allowed = None

# ======================
# CSV LOAD
# ======================
CSV_PATH = "whitelist_last4.csv"
df = pd.read_csv(CSV_PATH)
WHITELIST = set(df.iloc[:, 0].astype(str).tolist())

print(f"[INFO] CSV 로드 완료: {len(WHITELIST)} 개")

# ======================
# OCR UTILS
# ======================
def extract_last4(text):
    text = re.sub(r"\s+", "", text)
    m = re.search(r"(\d{4})$", text)
    return m.group(1) if m else None

def run_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5)
    text = pytesseract.image_to_string(gray, config=OCR_KOR)
    last4 = extract_last4(text)
    return text, last4

# ======================
# CAMERA
# ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] 시스템 시작")

set_led(yellow=True)

# ======================
# MAIN LOOP
# ======================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        distance_cm = ultrasonic.distance * 100

        # ----------------------
        # STATE: SCAN
        # ----------------------
        if state == "SCAN":
            set_led(yellow=True)
            cv2.putText(frame, "SCAN...", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)

            if distance_cm <= DIST_THRESHOLD_CM:
                if now - last_ocr_time >= OCR_INTERVAL:
                    print(f"[DEBUG] OCR EXECUTE (distance={distance_cm:.1f}cm)")
                    text, last4 = run_ocr(frame)
                    last_ocr_time = now

                    if last4:
                        last_digits = last4
                        last_allowed = last4 in WHITELIST
                        last_result_time = now
                        state = "RESULT"
                        print(f"[RESULT] {last4} → {'허용' if last_allowed else '미등록'}")

        # ----------------------
        # STATE: RESULT
        # ----------------------
        elif state == "RESULT":
            if last_allowed:
                set_led(green=True)
                label = f"ALLOW : {last_digits}"
                color = (0,255,0)
            else:
                set_led(red=True)
                label = f"DENY : {last_digits}"
                color = (0,0,255)

            cv2.putText(frame, label, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            if now - last_result_time >= RESULT_HOLD_SEC:
                state = "SCAN"
                last_digits = None
                last_allowed = None

        # ----------------------
        cv2.imshow("Parking OCR State Machine", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("종료")

finally:
    cap.release()
    cv2.destroyAllWindows()
    LED_RED.off()
    LED_GREEN.off()
    LED_YELLOW.off()
