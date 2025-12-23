#!/usr/bin/env python3
"""
Raspberry Pi 5
초음파(30cm) 트리거 → 번호판 OCR(뒷4자리 완성까지 대기) → 10초 타임아웃 시 미허용
LED: 기본 YELLOW, 허용 GREEN, 미허용 RED
"""

import cv2
import pytesseract
import pandas as pd
import time
import re
from gpiozero import DistanceSensor, LED

# =========================
# ===== 사용자 설정 =====
# =========================
CAM_INDEX = 0

DIST_THRESHOLD_CM = 30.0     # ★ 30cm 이하일 때만 스캔 시작
SCAN_TIMEOUT_SEC = 10.0      # ★ 스캔 시작 후 10초 안에 못 읽으면 미허용
OCR_INTERVAL = 0.8           # OCR 실행 주기(초) - 너무 짧으면 끊김 심해짐
RESULT_HOLD_SEC = 3.0        # 결과 표시 유지 시간
CSV_PATH = "whitelist_last4.csv"

# OCR 설정 (표시용/판단용 분리)
OCR_KOR = "--oem 3 --psm 6 -l kor+eng"
OCR_NUM = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

# =========================
# ===== GPIO 설정 =====
# =========================
ultra = DistanceSensor(trigger=17, echo=27, max_distance=2.0)

led_red = LED(23)
led_green = LED(12)
led_yellow = LED(20)

def set_led(r=False, g=False, y=False):
    led_red.on() if r else led_red.off()
    led_green.on() if g else led_green.off()
    led_yellow.on() if y else led_yellow.off()

# =========================
# ===== CSV 로드 =====
# =========================
def load_whitelist(csv_path):
    df = pd.read_csv(csv_path)
    col = "last4" if "last4" in df.columns else df.columns[0]
    return set(
        df[col].astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .dropna()
        .tolist()
    )

try:
    whitelist = load_whitelist(CSV_PATH)
except Exception as e:
    print("모르겠습니다: CSV 로드 실패:", e)
    whitelist = set()

# =========================
# ===== OCR 유틸 =====
# =========================
def extract_last4(text_num: str):
    t = re.sub(r"\s+", "", str(text_num))
    m = re.search(r"(\d{4})$", t)
    return m.group(1) if m else None

# =========================
# ===== 상태 정의 =====
# =========================
STATE_IDLE = 0      # 대기(노란불)
STATE_SCANNING = 1  # 30cm 이하 들어오면 10초 동안 OCR 시도
STATE_RESULT = 2    # 결과(초록/빨강) 유지 후 복귀

state = STATE_IDLE

scan_start_time = 0.0
last_ocr_time = 0.0

last4_digits = None
display_text = ""
allowed = False
result_time = 0.0

# =========================
# ===== 카메라 =====
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] 시작 - ESC 종료")
set_led(y=True)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        distance_cm = ultra.distance * 100

        # ---------- IDLE ----------
        if state == STATE_IDLE:
            set_led(y=True)
            cv2.putText(frame, f"IDLE  dist={distance_cm:.1f}cm",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.putText(frame, "SCAN WAIT (<=30cm)",
                        (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

            if distance_cm <= DIST_THRESHOLD_CM:
                # 스캔 시작
                state = STATE_SCANNING
                scan_start_time = now
                last_ocr_time = 0.0
                last4_digits = None
                display_text = ""
                print("[STATE] IDLE -> SCANNING")

        # ---------- SCANNING ----------
        elif state == STATE_SCANNING:
            set_led(y=True)

            elapsed = now - scan_start_time
            remain = max(0.0, SCAN_TIMEOUT_SEC - elapsed)

            cv2.putText(frame, f"SCANNING... {remain:.1f}s",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

            # 10초 타임아웃이면 미허용
            if elapsed >= SCAN_TIMEOUT_SEC:
                allowed = False
                display_text = "TIMEOUT (NO 4DIGITS)"
                result_time = now
                state = STATE_RESULT
                print("[RESULT] TIMEOUT -> DENY")
                continue

            # OCR 주기 제한
            if now - last_ocr_time >= OCR_INTERVAL:
                last_ocr_time = now
                print("[DEBUG] OCR EXECUTE")

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 판단용: 숫자만
                text_num = pytesseract.image_to_string(gray, config=OCR_NUM)
                last4 = extract_last4(text_num)

                # 표시용(한글 포함)은 성공했을 때만 한번 읽어도 됨(부하 감소)
                if last4:
                    text_kor = pytesseract.image_to_string(gray, config=OCR_KOR)
                    last4_digits = last4
                    allowed = last4_digits in whitelist
                    display_text = f"{text_kor.strip()} | last4={last4_digits}"
                    result_time = now
                    state = STATE_RESULT
                    print(f"[RESULT] {last4_digits} -> {'ALLOW' if allowed else 'DENY'}")

        # ---------- RESULT ----------
        elif state == STATE_RESULT:
            # LED 결과
            if allowed:
                set_led(g=True)
            else:
                set_led(r=True)

            label = "OK" if allowed else "DENY"
            color = (0,255,0) if allowed else (0,0,255)

            cv2.putText(frame, f"{label}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, display_text,
                        (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 일정 시간 유지 후 IDLE 복귀
            if now - result_time >= RESULT_HOLD_SEC:
                state = STATE_IDLE
                allowed = False
                display_text = ""
                last4_digits = None
                print("[STATE] RESULT -> IDLE")

        cv2.imshow("Parking OCR State Machine", frame)

        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    set_led(False, False, False)
