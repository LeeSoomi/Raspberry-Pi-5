#!/usr/bin/env python3
"""
Raspberry Pi 5 - Final Integrated
Ultrasonic trigger (<=30cm) -> ROI OCR -> backup full-frame OCR -> CSV(last4) compare -> LED
Timeout: 10s (no recognition => DENY)
"""

import cv2
import pytesseract
import pandas as pd
import time
import re
from gpiozero import DistanceSensor, LED

# =========================
# 사용자 설정
# =========================
CAM_INDEX = 0

DIST_THRESHOLD_CM = 30.0         # 초음파 트리거 거리
SCAN_TIMEOUT_SEC = 10.0          # 10초 동안만 인식 시도
OCR_INTERVAL_SEC = 0.6           # OCR 시도 간격(너무 짧으면 끊김 심해짐)
RESULT_HOLD_SEC = 3.0            # 결과 표시 유지

CSV_PATH = "whitelist_last4.csv" # last4 컬럼(또는 첫 컬럼)에 4자리 저장

# OCR 설정: 숫자만 인식(속도/정확도)
OCR_NUM = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

# ROI 탐색 파라미터(필요 시 완화 가능)
MIN_AREA = 1500                  # ROI 후보 최소 면적(너무 크면 검출 실패)
ASPECT_MIN, ASPECT_MAX = 2.0, 8.0

# 디버그 창
SHOW_DEBUG_ROI = True            # ROI 창 별도 표시(원하면 False)

# =========================
# GPIO (Pi5 권장: gpiozero + lgpio)
# =========================
ultra = DistanceSensor(trigger=17, echo=27, max_distance=2.0)

LED_RED = LED(23)
LED_GREEN = LED(12)
LED_YELLOW = LED(20)

def set_led(r=False, g=False, y=False):
    LED_RED.on() if r else LED_RED.off()
    LED_GREEN.on() if g else LED_GREEN.off()
    LED_YELLOW.on() if y else LED_YELLOW.off()

# =========================
# CSV 로드
# =========================
def load_whitelist_last4(csv_path: str) -> set:
    df = pd.read_csv(csv_path)
    col = "last4" if "last4" in df.columns else df.columns[0]
    # 문자열에서 4자리만 추출
    s = (
        df[col].astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .dropna()
        .str.strip()
    )
    return set(s.tolist())

try:
    WHITELIST = load_whitelist_last4(CSV_PATH)
    print(f"[INFO] CSV loaded: {len(WHITELIST)} items")
except Exception as e:
    print("모르겠습니다: CSV를 읽지 못했습니다.", e)
    WHITELIST = set()

# =========================
# OCR 유틸
# =========================
def extract_last4_from_text(text: str):
    """텍스트에서 마지막 4자리만 추출(정확히 4자리일 때만)"""
    t = re.sub(r"\s+", "", str(text))
    # 마지막 4자리
    m = re.search(r"(\d{4})$", t)
    return m.group(1) if m else None

def ocr_last4(img_bgr) -> str | None:
    """이미지(ROI 또는 전체)에서 last4 추출"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 전처리(속도/정확도 타협)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    txt = pytesseract.image_to_string(th, config=OCR_NUM)
    last4 = extract_last4_from_text(txt)

    # 혹시 OCR이 중간에 여러 숫자를 뱉으면 마지막 4자리 후보라도 찾기(백업)
    if not last4:
        digits = re.findall(r"\d{4}", re.sub(r"\s+", "", txt))
        if digits:
            last4 = digits[-1]
    return last4

# =========================
# ROI 찾기
# =========================
def find_plate_roi(frame_bgr):
    """번호판처럼 보이는 사각형 ROI 1개 반환(없으면 None)"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 80, 200)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_bbox = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        aspect = w / float(h)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        # 점수(면적 * 종횡비 안정성)로 "그럴듯한" 1개 선택
        score = area * (1.0 - abs(aspect - 4.0) / 4.0)
        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)
            best = approx

    if best_bbox is None:
        return None, None

    x, y, w, h = best_bbox
    # 약간 여백
    m = 6
    x2 = max(0, x - m)
    y2 = max(0, y - m)
    w2 = min(frame_bgr.shape[1] - x2, w + 2*m)
    h2 = min(frame_bgr.shape[0] - y2, h + 2*m)

    roi = frame_bgr[y2:y2+h2, x2:x2+w2]
    return roi, (x2, y2, w2, h2)

# =========================
# 상태 머신
# =========================
IDLE, SCANNING, RESULT = 0, 1, 2
state = IDLE

scan_start = 0.0
last_ocr_try = 0.0

result_time = 0.0
result_msg = ""
result_allowed = False
result_last4 = None

# =========================
# 카메라
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] ESC 종료")
set_led(y=True)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        dist_cm = ultra.distance * 100

        # ---------- IDLE ----------
        if state == IDLE:
            set_led(y=True)
            cv2.putText(frame, f"IDLE dist={dist_cm:.1f}cm (trigger<= {DIST_THRESHOLD_CM:.0f}cm)",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            cv2.putText(frame, "WAIT...",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

            if dist_cm <= DIST_THRESHOLD_CM:
                state = SCANNING
                scan_start = now
                last_ocr_try = 0.0
                result_msg = ""
                result_last4 = None
                print("[STATE] IDLE -> SCANNING")

        # ---------- SCANNING ----------
        elif state == SCANNING:
            remain = max(0.0, SCAN_TIMEOUT_SEC - (now - scan_start))
            cv2.putText(frame, f"SCANNING... {remain:.1f}s",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

            # 10초 타임아웃 => 인식 안됨(미허용)
            if (now - scan_start) >= SCAN_TIMEOUT_SEC:
                result_allowed = False
                result_last4 = None
                result_msg = "DENY: 인식 안됨 (timeout)"
                result_time = now
                state = RESULT
                print("[RESULT] TIMEOUT -> DENY")
                continue

            # OCR 시도 주기 제한
            if (now - last_ocr_try) >= OCR_INTERVAL_SEC:
                last_ocr_try = now

                # 1) ROI OCR 시도
                roi, bbox = find_plate_roi(frame)
                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

                    if SHOW_DEBUG_ROI and roi is not None and roi.size > 0:
                        cv2.imshow("DEBUG_ROI", roi)

                    last4 = ocr_last4(roi) if roi is not None else None
                    print(f"[DEBUG] ROI OCR -> {last4}")

                else:
                    last4 = None
                    if SHOW_DEBUG_ROI:
                        # ROI가 없는 경우 이전 ROI 창이 남는 걸 방지(선택)
                        pass

                # 2) ROI 실패 시 백업: 전체 프레임 OCR
                if not last4:
                    last4 = ocr_last4(frame)
                    print(f"[DEBUG] FULL OCR (backup) -> {last4}")

                # 3) 4자리 인식 성공 시 즉시 비교/결과
                if last4 and last4.isdigit() and len(last4) == 4:
                    result_last4 = last4
                    result_allowed = (last4 in WHITELIST)
                    result_msg = f"{'OK' if result_allowed else 'DENY'}: last4={last4}"
                    result_time = now
                    state = RESULT
                    print(f"[RESULT] {result_msg}")

        # ---------- RESULT ----------
        elif state == RESULT:
            # LED 표시
            if result_allowed:
                set_led(g=True)
                color = (0,255,0)
            else:
                set_led(r=True)
                color = (0,0,255)

            cv2.putText(frame, result_msg, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

            # 유지 후 IDLE 복귀
            if (now - result_time) >= RESULT_HOLD_SEC:
                state = IDLE
                result_msg = ""
                result_allowed = False
                result_last4 = None
                set_led(y=True)
                print("[STATE] RESULT -> IDLE")

        cv2.imshow("Parking OCR Final", frame)

        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    set_led(False, False, False)
