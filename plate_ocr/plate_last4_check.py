


#  plate_last4_check.py
import cv2
import numpy as np
import pandas as pd
import re
import time

# ---- OCR (Tesseract) 바인딩 ----
try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    print("확실하지 않음: pytesseract를 불러오지 못했습니다. `pip install pytesseract` 후 재시도하세요.")
    TESS_AVAILABLE = False

# ===== 사용자 설정 =====
CSV_PATH = "whitelist_last4.csv"  # CSV 파일 경로 (예: last4 컬럼에 4자리 숫자)
CAM_INDEX = 0                     # 웹캠 인덱스
SHOW_WINDOW = True                # 디버그 창 표시
MIN_AREA = 2500                   # 후보 사각형 최소 면적
ASPECT_RANGE = (2.5, 7.0)         # 번호판 가로/세로 비율 범위
TARGET_W, TARGET_H = 380, 80      # 원근보정 타겟 크기
CANNY1, CANNY2 = 80, 200          # 엣지 파라미터
BILATERAL = (9, 75, 75)           # 양방향 필터 파라미터
SUPPRESS_SEC = 3                  # 같은 결과 반복 출력 억제

# OCR 설정 (한 줄/여러 줄)
TESS_CONFIGS = [
    "--oem 3 --psm 7 -l kor+eng",
    "--oem 3 --psm 6 -l kor+eng"
]

# ===== CSV 로드: 4자리 숫자 화이트리스트 =====
def load_last4_whitelist(csv_path):
    df = pd.read_csv(csv_path)
    col = None
    # 우선순위: last4 컬럼 or 첫 번째 컬럼
    if "last4" in df.columns:
        col = "last4"
    else:
        col = df.columns[0]
    # 4자리만 필터링
    s = (
        df[col]
        .astype(str)
        .str.extract(r"(\d{4})", expand=False)  # 문자열 안의 첫 4자리 패턴
        .dropna()
        .str.strip()
    )
    return set(s.tolist())

# ===== 기하 유틸 =====
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상
    rect[2] = pts[np.argmax(s)]  # 우하
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상
    rect[3] = pts[np.argmax(diff)]  # 좌하
    return rect

def warp_plate(img, quad, w=TARGET_W, h=TARGET_H):
    rect = order_points(quad.reshape(4, 2).astype("float32"))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (w, h))

# ===== 전처리 & 후보 찾기 =====
def preprocess_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, 5
    )
    return th

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

# ===== OCR → 마지막 4자리 추출 =====
def ocr_texts(img_bgr):
    """여러 config로 OCR 시도 후 텍스트 리스트 반환"""
    if not TESS_AVAILABLE:
        return []
    out = []
    imgs = [preprocess_for_ocr(img_bgr), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)]
    for im in imgs:
        for cfg in TESS_CONFIGS:
            raw = pytesseract.image_to_string(im, config=cfg)
            if raw:
                out.append(raw)
    return out

def extract_last4_from_texts(texts):
    """
    다양한 문자열(예: '서울31바1234', '12가 3456')에서 마지막 4자리만 뽑아냄.
    여러 후보가 나오면 중복 제거.
    """
    results = []
    for t in texts:
        # 흔한 오인 교정
        t2 = t.replace("|", "1").replace("I", "1").replace("O", "0")
        # 줄바꿈/공백 제거
        t2 = re.sub(r"\s+", "", t2)

        # 1) 마지막 4자리 숫자
        m = re.search(r"(\d{4})$", t2)
        if m:
            results.append(m.group(1))
            continue

        # 2) 문자열 어디에든 4자리 숫자가 있을 때(예외적)
        m2 = re.findall(r"\d{4}", t2)
        if m2:
            results.extend(m2[-1:])  # 가장 마지막 것 하나만

    # 중복 제거
    uniq = []
    for v in results:
        if v not in uniq:
            uniq.append(v)
    return uniq

# ===== 메인 루프 =====
def main():
    # 화이트리스트 로드
    try:
        whitelist = load_last4_whitelist(CSV_PATH)
    except Exception as e:
        print("모르겠습니다: CSV를 읽는 중 오류가 발생했습니다:", e)
        return

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("모르겠습니다: 웹캠을 열 수 없습니다. CAM_INDEX를 확인하세요.")
        return

    # 해상도 (권장)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] ESC 종료. 인식된 '마지막 4자리'와 화이트리스트 대조 결과를 출력합니다.")
    last_print = {}  # { '3456': timestamp }

    while True:
        ok, frame = cap.read()
        if not ok:
            print("확실하지 않음: 프레임 캡처 실패.")
            break

        detected_any = False
        quads = find_plate_quads(frame)

        for q in quads:
            roi = warp_plate(frame, q, TARGET_W, TARGET_H)
            texts = ocr_texts(roi) if TESS_AVAILABLE else []
            last4s = extract_last4_from_texts(texts)

            if SHOW_WINDOW:
                x,y,w,h = cv2.boundingRect(q)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

            for last4 in last4s:
                now = time.time()
                if last4 not in last_print or (now - last_print[last4]) > SUPPRESS_SEC:
                    in_list = last4 in whitelist
                    print(f"[PLATE_LAST4] {last4} -> {'허용(리스트 포함)' if in_list else '미등록(리스트 불포함)'}")
                    last_print[last4] = now

                detected_any = True
                if SHOW_WINDOW:
                    color = (0,255,0) if last4 in whitelist else (0,0,255)
                    cx, cy = int(q[:,0,0].mean()), int(q[:,0,1].mean())
                    cv2.putText(frame, last4, (cx-40, cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # 후보가 전혀 없을 때: 전체 프레임으로 보조 시도 (원경/각도 불량 대비)
        if not detected_any and TESS_AVAILABLE:
            texts = ocr_texts(frame)
            last4s = extract_last4_from_texts(texts)
            for last4 in last4s:
                now = time.time()
                if last4 not in last_print or (now - last_print[last4]) > SUPPRESS_SEC:
                    in_list = last4 in whitelist
                    print(f"[FULL_LAST4] {last4} -> {'허용' if in_list else '미등록'}")
                    last_print[last4] = now
                if SHOW_WINDOW:
                    color = (0,255,0) if last4 in whitelist else (0,0,255)
                    cv2.putText(frame, last4, (30,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        if SHOW_WINDOW:
            cv2.imshow("Plate Last4 Checker", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
