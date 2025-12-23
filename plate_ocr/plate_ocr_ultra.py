#!/usr/bin/env python3
import cv2
import pytesseract
import pandas as pd
import time
import re
from gpiozero import DistanceSensor, LED

# ================= 사용자 설정 =================
CAM_INDEX = 0
DIST_THRESHOLD_CM = 30.0
SCAN_TIMEOUT = 10.0
CAPTURE_DELAY = 0.7
CAPTURE_COUNT = 3
RESULT_HOLD = 3.0

CSV_PATH = "plates.csv"
OCR_KOR = "--oem 3 --psm 6 -l kor+eng"
OCR_NUM = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

# ================= GPIO =================
ultra = DistanceSensor(trigger=17, echo=27, max_distance=2.0)
LED_R = LED(23)
LED_G = LED(12)
LED_Y = LED(20)

def set_led(r=False, g=False, y=False):
    LED_R.on() if r else LED_R.off()
    LED_G.on() if g else LED_G.off()
    LED_Y.on() if y else LED_Y.off()

# ================= CSV 로드 =================
def load_whitelist(csv_path):
    df = pd.read_csv(csv_path)
    col = "plate" if "plate" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip())

WHITELIST = load_whitelist(CSV_PATH)

# ================= 번호판 파싱 =================
def parse_plate(text):
    t = re.sub(r"\s+", "", str(text))

    m = re.search(r'(\d{2,3})([가-힣])(\d{4})', t)
    if m:
        return {"full": m.group(0), "last4": m.group(3), "confidence": "HIGH"}

    m = re.search(r'[가-힣](\d{4})', t)
    if m:
        return {"full": None, "last4": m.group(1), "confidence": "MID"}

    m = re.search(r'(\d{4})$', t)
    if m:
        return {"full": None, "last4": m.group(1), "confidence": "LOW"}

    return None

def match_plate(parsed, whitelist):
    if parsed["full"] and parsed["full"] in whitelist:
        return True
    for p in whitelist:
        if p.endswith(parsed["last4"]):
            return True
    return False

# ================= ROI 탐색 =================
def find_plate_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.bilateralFilter(gray,9,75,75), 80, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < 1500:
            continue
        x,y,w,h = cv2.boundingRect(c)
        if 2.0 < w/h < 8.0:
            return frame[y:y+h, x:x+w]
    return None

def ocr_image(img, cfg):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return pytesseract.image_to_string(th, config=cfg)

# ================= 메인 =================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3,1280)
cap.set(4,720)

state = "IDLE"
scan_start = 0
result_time = 0
result_text = ""
result_allowed = False

set_led(y=True)
print("[SYSTEM] Parking OCR Stable Final - ESC 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    now = time.time()
    dist = ultra.distance * 100

    if state == "IDLE":
        set_led(y=True)
        cv2.putText(frame, f"IDLE {dist:.1f}cm", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        if dist <= DIST_THRESHOLD_CM:
            state = "SCAN"
            scan_start = now
            time.sleep(CAPTURE_DELAY)

    elif state == "SCAN":
        cv2.putText(frame, "SCANNING...", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        if now - scan_start > SCAN_TIMEOUT:
            result_text = "DENY: 인식 안됨"
            result_allowed = False
            result_time = now
            state = "RESULT"
        else:
            for _ in range(CAPTURE_COUNT):
                ret, f = cap.read()
                if not ret:
                    continue

                roi = find_plate_roi(f)
                text = ""

                if roi is not None:
                    text = ocr_image(roi, OCR_KOR)

                if not text.strip():
                    text = ocr_image(f, OCR_KOR)

                parsed = parse_plate(text)
                if parsed:
                    allowed = match_plate(parsed, WHITELIST)
                    result_text = f"{parsed['last4']} ({parsed['confidence']})"
                    result_allowed = allowed
                    result_time = now
                    state = "RESULT"
                    break

    elif state == "RESULT":
        set_led(g=result_allowed, r=not result_allowed)
        cv2.putText(frame, result_text, (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,
                    (0,255,0) if result_allowed else (0,0,255),3)

        if now - result_time > RESULT_HOLD:
            state = "IDLE"

    cv2.imshow("Parking OCR Stable", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
set_led(False,False,False)
