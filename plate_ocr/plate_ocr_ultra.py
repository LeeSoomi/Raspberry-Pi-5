#!/usr/bin/env python3
"""
Raspberry Pi 5
Ultrasonic trigger (30cm) → Plate ROI OCR → last4 compare → LED result
"""

import cv2
import pytesseract
import pandas as pd
import time
import re
from gpiozero import DistanceSensor, LED

# =========================
# 설정
# =========================
CAM_INDEX = 0
DIST_THRESHOLD = 30.0
SCAN_TIMEOUT = 10.0
OCR_INTERVAL = 0.8
RESULT_HOLD = 3.0

CSV_PATH = "whitelist_last4.csv"

# GPIO
ultra = DistanceSensor(trigger=17, echo=27, max_distance=2.0)
LED_RED = LED(23)
LED_GREEN = LED(12)
LED_YELLOW = LED(20)

def set_led(r=False, g=False, y=False):
    LED_RED.on() if r else LED_RED.off()
    LED_GREEN.on() if g else LED_GREEN.off()
    LED_YELLOW.on() if y else LED_YELLOW.off()

# OCR 설정
OCR_NUM = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

# =========================
# CSV 로드
# =========================
def load_csv(path):
    df = pd.read_csv(path)
    col = "last4" if "last4" in df.columns else df.columns[0]
    return set(
        df[col].astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .dropna()
        .tolist()
    )

WHITELIST = load_csv(CSV_PATH)

# =========================
# 번호판 ROI 검출
# =========================
def find_plate_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 80, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2500:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h)
            if 2.5 < aspect < 6.5:
                return frame[y:y+h, x:x+w], (x, y, w, h)

    return None, None

# =========================
# OCR
# =========================
def extract_last4(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(th, config=OCR_NUM)
    text = re.sub(r"\s+", "", text)

    m = re.search(r"(\d{4})$", text)
    return m.group(1) if m else None

# =========================
# 상태
# =========================
IDLE, SCAN, RESULT = 0, 1, 2
state = IDLE

scan_start = 0
last_ocr = 0
result_time = 0

last4 = None
allowed = False
msg = ""

# =========================
# 카메라
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3, 1280)
cap.set(4, 720)

set_led(y=True)

print("[START] Parking OCR System")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        dist = ultra.distance * 100

        # ---------- IDLE ----------
        if state == IDLE:
            set_led(y=True)
            cv2.putText(frame, f"IDLE  dist={dist:.1f}cm",
                        (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            if dist <= DIST_THRESHOLD:
                state = SCAN
                scan_start = now
                last_ocr = 0
                last4 = None
                print("[STATE] SCAN START")

        # ---------- SCAN ----------
        elif state == SCAN:
            remain = SCAN_TIMEOUT - (now - scan_start)
            cv2.putText(frame, f"SCAN... {remain:.1f}s",
                        (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            if now - scan_start >= SCAN_TIMEOUT:
                msg = "인식 안됨"
                allowed = False
                state = RESULT
                result_time = now
                print("[RESULT] TIMEOUT")

            if now - last_ocr >= OCR_INTERVAL:
                last_ocr = now
                roi, bbox = find_plate_roi(frame)
                if roi is not None:
                    last4 = extract_last4(roi)
                    if last4:
                        allowed = last4 in WHITELIST
                        msg = f"{last4} {'허용' if allowed else '미허용'}"
                        state = RESULT
                        result_time = now
                        print(f"[RESULT] {msg}")

        # ---------- RESULT ----------
        elif state == RESULT:
            set_led(g=allowed, r=not allowed)
            cv2.putText(frame, msg,
                        (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0,255,0) if allowed else (0,0,255), 3)

            if now - result_time >= RESULT_HOLD:
                state = IDLE
                msg = ""
                last4 = None
                set_led(y=True)

        cv2.imshow("Parking OCR ROI", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    set_led(False, False, False)
