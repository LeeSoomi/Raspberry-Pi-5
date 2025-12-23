#!/usr/bin/env python3
import cv2, pytesseract, pandas as pd, time, re
from gpiozero import DistanceSensor, LED

# ================= 설정 =================
CAM_INDEX = 0
DIST_THRESHOLD_CM = 30.0
SCAN_TIMEOUT = 10.0
CAPTURE_DELAY = 0.7      # 차량 정지 대기
CAPTURE_COUNT = 3        # 캡처 장수
RESULT_HOLD = 3.0

CSV_PATH = "whitelist_last4.csv"
OCR_NUM = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
OCR_KOR = "--oem 3 --psm 6 -l kor+eng"

# ================= GPIO =================
ultra = DistanceSensor(trigger=17, echo=27, max_distance=2.0)
LED_R, LED_G, LED_Y = LED(23), LED(12), LED(20)

def set_led(r=False, g=False, y=False):
    LED_R.on() if r else LED_R.off()
    LED_G.on() if g else LED_G.off()
    LED_Y.on() if y else LED_Y.off()

# ================= CSV =================
def load_last4(csv):
    df = pd.read_csv(csv)
    col = "last4" if "last4" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.extract(r'(\d{4})', expand=False).dropna())

WHITELIST = load_last4(CSV_PATH)

# ================= 번호판 파싱 =================
def parse_plate(text):
    t = re.sub(r"\s+", "", str(text))
    m = re.search(r'(\d{2,3})([가-힣])(\d{4})', t)
    if m:
        return {"last4": m.group(3), "confidence": "HIGH"}
    m = re.search(r'[가-힣](\d{4})', t)
    if m:
        return {"last4": m.group(1), "confidence": "MID"}
    m = re.search(r'(\d{4})$', t)
    if m:
        return {"last4": m.group(1), "confidence": "LOW"}
    return None

# ================= ROI =================
def find_plate_roi(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.bilateralFilter(g,9,75,75),80,200)
    cnts,_ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 1500: continue
        x,y,w,h = cv2.boundingRect(c)
        if 2.0 < w/h < 8.0:
            return frame[y:y+h, x:x+w]
    return None

def ocr_text(img, cfg):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return pytesseract.image_to_string(th, config=cfg)

# ================= 메인 =================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3,1280); cap.set(4,720)

set_led(y=True)
state = "IDLE"
scan_start = 0

print("[START] ESC 종료")

while True:
    ret, frame = cap.read()
    if not ret: continue
    now = time.time()
    dist = ultra.distance * 100

    if state == "IDLE":
        set_led(y=True)
        cv2.putText(frame, f"IDLE {dist:.1f}cm", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        if dist <= DIST_THRESHOLD_CM:
            scan_start = now
            state = "SCAN"
            time.sleep(CAPTURE_DELAY)

    elif state == "SCAN":
        if now - scan_start > SCAN_TIMEOUT:
            result = "인식 안됨"
            allowed = False
            state = "RESULT"
            result_time = now
        else:
            for _ in range(CAPTURE_COUNT):
                ret, f = cap.read()
                if not ret: continue
                roi = find_plate_roi(f)
                text = ""
                if roi is not None:
                    text = ocr_text(roi, OCR_KOR)
                if not text.strip():
                    text = ocr_text(f, OCR_KOR)

                parsed = parse_plate(text)
                if parsed:
                    last4 = parsed["last4"]
                    allowed = last4 in WHITELIST
                    result = f"{last4} ({parsed['confidence']})"
                    state = "RESULT"
                    result_time = now
                    break

    elif state == "RESULT":
        set_led(g=allowed, r=not allowed)
        cv2.putText(frame, result, (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,
                    (0,255,0) if allowed else (0,0,255),3)
        if now - result_time > RESULT_HOLD:
            state = "IDLE"

    cv2.imshow("Parking OCR Capture", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
set_led(False,False,False)
