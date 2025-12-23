
# 설치체크
# sudo apt update
# sudo apt install -y tesseract-ocr tesseract-ocr-kor python3-opencv python3-pandas
# python3 -m pip install pytesseract --break-system-packages


import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
import time
from enum import Enum

# ===================== 사용자 설정 =====================
CAM_INDEX = 0

# 카메라 입력 해상도(가능하면 1280x720)
CAP_W, CAP_H = 1280, 720

# 표시 창 크기(표시만 줄여서 부드럽게)
SHOW_W, SHOW_H = 960, 540

# OCR 주기 제한 (초)
OCR_INTERVAL_SEC = 1.0          # 숫자 판단 OCR 주기 (빠름)
KOR_INTERVAL_SEC = 2.5          # 한글 포함 표시 OCR 주기 (느림)

# 결과 표시 유지 시간
RESULT_HOLD_SEC = 3.0           # GREEN/RED 유지 후 다시 대기

# 거리 트리거 (초음파 연동 전 임시: True면 항상 OCR 시도)
ENABLE_TRIGGER = True

# CSV (마지막 4자리)
CSV_PATH = "whitelist_last4.csv"    # 컬럼: last4 또는 첫 컬럼

# 번호판 후보 영역 찾기 파라미터
MIN_AREA = 2500
ASPECT_RANGE = (2.5, 7.0)
TARGET_W, TARGET_H = 420, 110
CANNY1, CANNY2 = 80, 200
BILATERAL = (9, 75, 75)

# ===================== OCR 설정 분리 =====================
# 판단용(숫자만): 빠르고 안정적
OCR_NUM = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
# 표시용(한글 포함): 느리지만 가끔만 수행
OCR_KOR = "--oem 3 --psm 6 -l kor+eng"

# ===================== 유틸: CSV 로드 =====================
def load_last4_whitelist(csv_path):
    df = pd.read_csv(csv_path)
    if "last4" in df.columns:
        col = "last4"
    else:
        col = df.columns[0]
    s = (
        df[col].astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .dropna()
        .str.strip()
    )
    return set(s.tolist())

# ===================== 유틸: 기하/워핑 =====================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_plate(img, quad, w=TARGET_W, h=TARGET_H):
    rect = order_points(quad.reshape(4, 2).astype("float32"))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (w, h))

# ===================== 전처리 =====================
def preprocess_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 너무 강한 이진화는 한글에 불리. 숫자판단은 그레이 그대로도 충분.
    # 약한 샤프닝 + 적당한 블러
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    return gray, blur

# ===================== 후보 찾기(번호판 사각형) =====================
def find_plate_quads(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, *BILATERAL)
    edges = cv2.Canny(blur, CANNY1, CANNY2)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x,y,w,h = cv2.boundingRect(approx)
            aspect = w / float(h)
            if ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]:
                quads.append(approx)
    return quads

# ===================== OCR/추출 =====================
def extract_last4_from_digits_text(t):
    t2 = re.sub(r"\s+", "", str(t))
    # 숫자만 OCR 했더라도 혹시 섞인 경우 대비
    nums = re.findall(r"\d+", t2)
    if not nums:
        return None
    joined = "".join(nums)
    if len(joined) < 4:
        return None
    return joined[-4:]

def ocr_num_last4(roi_bgr):
    gray, blur = preprocess_for_ocr(roi_bgr)
    # 숫자 판단은 blur를 사용(노이즈 줄이기)
    txt = pytesseract.image_to_string(blur, config=OCR_NUM)
    return extract_last4_from_digits_text(txt)

def ocr_kor_full(roi_bgr):
    gray, blur = preprocess_for_ocr(roi_bgr)
    # 표시용은 gray 기준
    txt = pytesseract.image_to_string(gray, config=OCR_KOR)
    txt = re.sub(r"\s+", "", txt)
    # 너무 긴/이상한 문자열 컷
    if len(txt) > 20:
        txt = txt[:20]
    return txt

# ===================== 상태 머신 =====================
class State(Enum):
    IDLE = 0          # 기본 대기 (YELLOW)
    SCAN = 1          # OCR 스캔 중
    HOLD_RESULT = 2   # 결과 유지 (GREEN/RED)

def main():
    # CSV 로드
    try:
        whitelist = load_last4_whitelist(CSV_PATH)
        print(f"[INFO] whitelist loaded: {len(whitelist)}")
    except Exception as e:
        print("모르겠습니다: CSV 로드 실패:", e)
        return

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("모르겠습니다: 카메라를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    # 실제 적용 해상도 확인
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera actual: {real_w}x{real_h}")

    state = State.IDLE
    last_num_ocr_t = 0.0
    last_kor_ocr_t = 0.0
    hold_until = 0.0

    # 화면 표시용 상태값
    last4 = None
    full_text = None
    allowed = None

    print("[INFO] q=quit, r=reload csv")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()

        # 표시용 프레임은 리사이즈(부드럽게)
        display = cv2.resize(frame, (SHOW_W, SHOW_H))

        # ========== 트리거 조건 ==========
        # 초음파 붙이면 여기에서 distance < threshold 일 때만 SCAN으로 전환하면 됨
        trigger_ok = ENABLE_TRIGGER

        # ========== 상태 전이 ==========
        if state == State.IDLE:
            # 기본 표시 (노란 상태)
            last4, full_text, allowed = None, None, None

            if trigger_ok:
                state = State.SCAN

        elif state == State.SCAN:
            # OCR은 주기 제한
            do_num_ocr = (now - last_num_ocr_t) >= OCR_INTERVAL_SEC
            do_kor_ocr = (now - last_kor_ocr_t) >= KOR_INTERVAL_SEC

            if do_num_ocr or do_kor_ocr:
                last_num_ocr_t = now if do_num_ocr else last_num_ocr_t
                last_kor_ocr_t = now if do_kor_ocr else last_kor_ocr_t

                # 후보 영역 찾기(원본 frame 기준)
                quads = find_plate_quads(frame)

                # 가장 큰 후보 하나만 처리(속도용)
                best_q = None
                best_area = 0
                for q in quads:
                    x,y,w,h = cv2.boundingRect(q)
                    area = w*h
                    if area > best_area:
                        best_area = area
                        best_q = q

                if best_q is not None:
                    roi = warp_plate(frame, best_q)

                    # 숫자 판단용 OCR
                    if do_num_ocr:
                        got_last4 = ocr_num_last4(roi)
                        if got_last4 and len(got_last4) == 4:
                            last4 = got_last4
                            allowed = (last4 in whitelist)

                            # 결과가 나오면 HOLD_RESULT로
                            hold_until = now + RESULT_HOLD_SEC
                            state = State.HOLD_RESULT

                    # 표시용 OCR (한글 포함) - 결과 유지용 텍스트
                    if do_kor_ocr:
                        full_text = ocr_kor_full(roi)

        elif state == State.HOLD_RESULT:
            # 결과 유지 시간이 지나면 IDLE로 복귀
            if now >= hold_until:
                state = State.IDLE

        # ========== 화면 오버레이(끊김 최소: 가벼운 작업만) ==========
        # 상태 텍스트
        if state == State.IDLE:
            status = "IDLE (YELLOW)"
        elif state == State.SCAN:
            status = "SCAN..."
        else:
            status = "RESULT HOLD"

        cv2.putText(display, status, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # OCR 결과 표시
        if full_text:
            cv2.putText(display, f"FULL: {full_text}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        if last4:
            msg = f"LAST4: {last4}  ->  {'ALLOW' if allowed else 'DENY'}"
            color = (0,255,0) if allowed else (0,0,255)
            cv2.putText(display, msg, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Parking OCR State Machine", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            try:
                whitelist = load_last4_whitelist(CSV_PATH)
                print(f"[INFO] whitelist reloaded: {len(whitelist)}")
            except Exception as e:
                print("확실하지 않음: CSV 재로드 실패:", e)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
