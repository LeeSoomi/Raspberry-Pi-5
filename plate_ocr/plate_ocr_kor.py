# plate
# 12가3456
# 123나4567
# 서울31바1234




import cv2
import numpy as np
import re
import time

# ---- OCR(테서랙트) 사용 ----
# pytesseract는 파이썬 바인딩입니다. 시스템에 tesseract-ocr, tesseract-ocr-kor가 설치되어 있어야 합니다.
try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception as e:
    print("확실하지 않음: pytesseract를 불러오지 못했습니다. `pip install pytesseract` 필요.")
    TESS_AVAILABLE = False

# ===== 사용자 조정 파라미터 =====
CAM_INDEX = 0                 # 웹캠 인덱스
SHOW_DEBUG = True             # 중간 처리 화면 보기
MIN_AREA = 2500               # 너무 작은 후보 제거(픽셀)
ASPECT_RANGE = (2.5, 7.0)     # 번호판 가로/세로 비율 범위
TARGET_W, TARGET_H = 380, 80  # 원근보정 후 타겟 크기
CANNY1, CANNY2 = 80, 200      # 엣지 파라미터
BILATERAL = (9, 75, 75)

# 한국 번호판 정규식(구형/신형 혼합 대응)
PATTERNS = [
    re.compile(r"(\d{2,3}\s?[가-힣]\s?\d{4})"),            # 12가3456, 123나4567
    re.compile(r"([가-힣]{2}\s?\d{2,3}\s?[가-힣]\s?\d{4})") # 서울31바1234
]

# Tesseract 설정 후보(단일 라인/여러 라인)
TESS_CONFIGS = [
    "--oem 3 --psm 7 -l kor+eng",   # 한 줄 가정
    "--oem 3 --psm 6 -l kor+eng"    # 여러 줄
]

def order_points(pts):
    # 4점(사각형)을 좌상, 우상, 우하, 좌하로 정렬
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
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped

def preprocess_for_ocr(img_bgr):
    # 그레이 → 가우시안/양방향 → 적응형 임계
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, 5
    )
    return th

def extract_plates_from_text(text):
    # 공백 제거 및 오인 교정
    def norm(s):
        s = s.replace(" ", "").replace("|", "1").replace("I", "1").replace("O", "0")
        return s
    results = []
    for pat in PATTERNS:
        for m in pat.findall(text):
            results.append(norm(m))
    # 중복 제거(순서 유지)
    seen, uniq = set(), []
    for r in results:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq

def ocr_plate(img_bgr):
    if not TESS_AVAILABLE:
        return []

    candidates = []
    # 전처리된 이진/그레이 둘 다 시도
    pre = preprocess_for_ocr(img_bgr)
    imgs = [pre, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)]
    for im in imgs:
        for cfg in TESS_CONFIGS:
            raw = pytesseract.image_to_string(im, config=cfg)
            plates = extract_plates_from_text(raw)
            candidates.extend(plates)
    # 최종 중복 제거
    uniq = []
    for p in candidates:
        if p not in uniq:
            uniq.append(p)
    return uniq

def find_plate_quads(frame_bgr):
    """번호판처럼 보이는 사각형 후보(근사 사각형)를 찾고 쿼드 반환"""
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

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("모르겠습니다: 웹캠을 열 수 없습니다. CAM_INDEX를 확인하세요.")
        return

    # 해상도(선택)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] ESC 종료. 번호판 인식 시 콘솔과 화면에 표시합니다.")
    last_print = {}
    SUPPRESS_SEC = 3  # 같은 텍스트 반복 출력 억제

    while True:
        ok, frame = cap.read()
        if not ok:
            print("확실하지 않음: 프레임 캡처 실패.")
            break

        quads = find_plate_quads(frame)

        detected_any = False
        for q in quads:
            plate_roi = warp_plate(frame, q, TARGET_W, TARGET_H)
            plates = ocr_plate(plate_roi)

            if SHOW_DEBUG:
                x,y,w,h = cv2.boundingRect(q)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

            for text in plates:
                now = time.time()
                if text not in last_print or (now - last_print[text]) > SUPPRESS_SEC:
                    print(f"[PLATE] {text}")
                    last_print[text] = now
                detected_any = True
                if SHOW_DEBUG:
                    # 번호판 영역 위에 텍스트 오버레이
                    cx, cy = int(q[:,0,0].mean()), int(q[:,0,1].mean())
                    cv2.putText(frame, text, (cx-80, cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        # 후보가 전혀 없을 때(원경/각도 불량 등) 전체 프레임 OCR로 보조 시도
        if not detected_any:
            if TESS_AVAILABLE:
                gray_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for cfg in TESS_CONFIGS:
                    raw = pytesseract.image_to_string(gray_small, config=cfg)
                    plates = extract_plates_from_text(raw)
                    for t in plates:
                        now = time.time()
                        if t not in last_print or (now - last_print[t]) > SUPPRESS_SEC:
                            print(f"[FULL] {t}")
                            last_print[t] = now
                        if SHOW_DEBUG:
                            cv2.putText(frame, t, (30,60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2, cv2.LINE_AA)

        if SHOW_DEBUG:
            cv2.imshow("Korean Plate OCR (OpenCV + Tesseract)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
