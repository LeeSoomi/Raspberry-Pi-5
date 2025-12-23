# 상세 주석 내용 추가

import cv2                 # OpenCV: 카메라 프레임 획득/영상 처리/화면 출력
import pytesseract         # Tesseract OCR 파이썬 바인딩: 이미지 → 텍스트 인식
import pandas as pd        # CSV 로드/처리용 (plates.csv를 편하게 읽기)
import time                # 시간 측정/딜레이(대기)용
import re                  # 정규표현식: OCR 결과에서 번호판 패턴 추출
from gpiozero import DistanceSensor, LED
# gpiozero: Raspberry Pi 5에서도 안정적으로 GPIO 제어(초음파/LED)

# ================= 사용자 설정 =================
CAM_INDEX = 0              # 카메라 장치 번호 (보통 0이 첫 번째 카메라)
DIST_THRESHOLD_CM = 30.0   # 초음파로 측정된 거리가 30cm 이하이면 스캔 시작
SCAN_TIMEOUT = 10.0        # 스캔 시작 후 10초 동안만 번호판 인식 시도
CAPTURE_DELAY = 0.7        # 초음파 트리거 직후 0.7초 대기(차량 정지/초점 안정 목적)
CAPTURE_COUNT = 3          # 스캔 중 프레임을 3장 읽고 OCR 시도(3번 기회)
RESULT_HOLD = 3.0          # 결과(허용/미허용)를 화면/LED로 3초 동안 유지

CSV_PATH = "plates.csv"    # 허용된 차량 번호판 목록이 들어있는 CSV 파일
OCR_KOR = "--oem 3 --psm 6 -l kor+eng"
# OCR_KOR:
#  - oem 3: Tesseract 엔진 자동(기본 LSTM 포함)
#  - psm 6: 문단/블록 형태로 인식(번호판+한글 포함 문자열 인식에 사용)
#  - -l kor+eng: 한글+영어(숫자 포함) 인식

OCR_NUM = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
# OCR_NUM:
#  - psm 7: 한 줄(한 개 라인)로 인식
#  - whitelist=0123456789: 숫자만 인식하도록 제한(속도/정확도 개선 목적)
# ※ 현재 코드에서는 OCR_NUM 변수를 선언해두었지만 실제로는 OCR_KOR만 사용합니다.

# ================= GPIO =================
# 초음파 센서: TRIG=GPIO17, ECHO=GPIO27
# gpiozero DistanceSensor는 내부적으로 타이밍을 처리해서 거리(distance)를 0~1 범위로 제공
# (max_distance=2.0은 최대 2m까지 측정 범위를 의미)
ultra = DistanceSensor(trigger=17, echo=27, max_distance=2.0)

# LED 핀 설정(BCM 번호)
LED_R = LED(23)  # RED: GPIO 23
LED_G = LED(12)  # GREEN: GPIO 12
LED_Y = LED(20)  # YELLOW: GPIO 20

def set_led(r=False, g=False, y=False):
    """
    LED 상태를 한 번에 제어하는 함수
    - r=True면 빨간 LED 켬, False면 끔
    - g=True면 초록 LED 켬, False면 끔
    - y=True면 노랑 LED 켬, False면 끔
    """
    LED_R.on() if r else LED_R.off()
    LED_G.on() if g else LED_G.off()
    LED_Y.on() if y else LED_Y.off()

# ================= CSV 로드 =================
def load_whitelist(csv_path):
    """
    CSV 파일에서 허용 번호판 목록을 읽어 set으로 반환
    - CSV 컬럼 이름이 'plate'이면 그 컬럼 사용
    - 없으면 첫 번째 컬럼 사용
    - strip()으로 앞뒤 공백 제거
    - set으로 만들어 검색(포함 여부 판단)을 빠르게 함
    """
    df = pd.read_csv(csv_path)
    col = "plate" if "plate" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip())

# 허용 번호판 목록을 프로그램 시작 시 1회 로드
WHITELIST = load_whitelist(CSV_PATH)

# ================= 번호판 파싱 =================
def parse_plate(text):
    """
    OCR 결과 문자열(text)에서 번호판 구성 요소를 뽑아내는 함수

    한국 번호판 예:
      152가3018
      12나3456
      345다7890

    처리 순서(신뢰도 높은 것부터):
    1) (숫자2~3)(한글1)(숫자4) 구조로 완전 매칭되면 HIGH
    2) 한글 뒤에 (숫자4)만이라도 있으면 MID
    3) 문자열 끝이 (숫자4)면 LOW (최후 보험)
    """
    # OCR은 공백/줄바꿈이 섞여 나오기 쉬워서 공백류 제거
    t = re.sub(r"\s+", "", str(text))

    # 1) 정식 번호판 구조: 앞자리2~3 + 한글 + 뒷자리4
    # 예: 152가3018, 12나3456, 345다7890
    m = re.search(r'(\d{2,3})([가-힣])(\d{4})', t)
    if m:
        # full: 전체 번호판 문자열(예: 152가3018)
        # last4: 뒷자리 4개(예: 3018)
        # confidence: "HIGH" (가장 신뢰)
        return {"full": m.group(0), "last4": m.group(3), "confidence": "HIGH"}

    # 2) 한글 + 뒷자리4 형태라도 있으면 MID
    # 예: "...가3018" 같은 부분만 인식된 경우
    m = re.search(r'[가-힣](\d{4})', t)
    if m:
        # full은 없고 last4만 확보한 상태
        return {"full": None, "last4": m.group(1), "confidence": "MID"}

    # 3) 마지막이 숫자4면 LOW (정말 안 될 때 최소 판단)
    m = re.search(r'(\d{4})$', t)
    if m:
        return {"full": None, "last4": m.group(1), "confidence": "LOW"}

    # 어떤 조건도 못 맞추면 None 반환(번호판으로 볼 수 없음)
    return None

def match_plate(parsed, whitelist):
    """
    파싱 결과(parsed)가 whitelist(허용 목록)에 포함되는지 판단

    우선순위:
    1) parsed["full"]이 있으면 전체 번호판으로 정확 비교 (가장 안전)
    2) full이 없으면 last4로 보조 비교 (끝 4자리 동일이면 허용 처리)
       - last4 방식은 중복 가능성이 있어 full보다 덜 안전함
    """
    # 1) 전체 번호판 비교
    if parsed["full"] and parsed["full"] in whitelist:
        return True

    # 2) last4로 보조 비교 (whitelist 항목 중 끝 4자리 동일한 것이 있으면 True)
    for p in whitelist:
        if p.endswith(parsed["last4"]):
            return True

    return False

# ================= ROI 탐색 =================
def find_plate_roi(frame):
    """
    frame(컬러 이미지)에서 번호판처럼 보이는 ROI 영역을 찾아 잘라 반환
    - 간단한 엣지 기반/윤곽 기반 접근
    - 완벽한 번호판 검출기(DNN)보다 가볍지만, 각도/조명에 따라 실패할 수 있음

    로직:
    1) 그레이스케일 변환
    2) bilateralFilter로 노이즈 줄이면서 엣지는 유지
    3) Canny로 엣지 검출
    4) contour(윤곽) 찾기
    5) 면적이 충분하고(>1500) 가로세로비(2~8)면 번호판 후보로 간주
    6) 후보 하나를 찾으면 그 영역을 잘라서 ROI 반환
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 엣지에 도움되는 필터 + Canny 엣지 검출
    edges = cv2.Canny(cv2.bilateralFilter(gray, 9, 75, 75), 80, 200)

    # 윤곽선 검출
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # 너무 작은 윤곽은 제외
        if cv2.contourArea(c) < 1500:
            continue

        # 윤곽을 감싸는 사각형 박스 추출
        x, y, w, h = cv2.boundingRect(c)

        # 번호판은 가로로 긴 편: w/h 비율이 특정 범위에 들어야 후보로 인정
        if 2.0 < w / h < 8.0:
            # 후보를 찾으면 즉시 ROI 반환(첫 후보)
            return frame[y:y+h, x:x+w]

    # 후보 없으면 None
    return None

def ocr_image(img, cfg):
    """
    이미지(img)를 OCR 수행하여 문자열 반환
    - 그레이스케일 → Otsu 이진화 → pytesseract.image_to_string
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(th, config=cfg)

# ================= 메인 =================
# 카메라 열기
cap = cv2.VideoCapture(CAM_INDEX)

# 카메라 해상도 설정(가능한 범위 내에서 적용됨)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

# 상태 머신 변수
state = "IDLE"          # "IDLE" 대기 상태 / "SCAN" 인식 시도 / "RESULT" 결과 표시
scan_start = 0          # 스캔 시작 시각(타임아웃 계산용)
result_time = 0         # 결과 표시 시작 시각(RESULT_HOLD 유지용)
result_text = ""        # 화면에 표시할 결과 문자열
result_allowed = False  # 허용(True)/미허용(False)

# 시작 상태는 노란불(대기)
set_led(y=True)
print("[SYSTEM] Parking OCR Stable Final - ESC 종료")

while True:
    # 프레임 한 장 읽기
    ret, frame = cap.read()
    if not ret:
        # 프레임 읽기 실패 시 다음 루프로
        continue

    now = time.time()

    # ultra.distance는 0~1 비율값(거리/최대거리)이므로 cm로 바꾸려면 *100
    # (max_distance=2.0이면 최대 200cm까지 표현되지만, 실제론 0~200cm 정도)
    dist = ultra.distance * 100

    # ================= IDLE 상태 =================
    if state == "IDLE":
        # 대기 상태: 노란불
        set_led(y=True)

        # 화면에 현재 거리 표시
        cv2.putText(frame, f"IDLE {dist:.1f}cm", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 초음파가 30cm 이하이면 스캔 상태로 전환
        if dist <= DIST_THRESHOLD_CM:
            state = "SCAN"
            scan_start = now

            # 차량이 멈추고 카메라 초점이 안정되도록 잠깐 대기
            time.sleep(CAPTURE_DELAY)

    # ================= SCAN 상태 =================
    elif state == "SCAN":
        # 스캔 중 화면 표시
        cv2.putText(frame, "SCANNING...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 스캔 시작 후 10초가 넘으면 타임아웃 처리
        if now - scan_start > SCAN_TIMEOUT:
            result_text = "DENY: 인식 안됨"
            result_allowed = False
            result_time = now
            state = "RESULT"
        else:
            # CAPTURE_COUNT만큼 프레임을 더 읽으며 OCR 시도(총 3회 기회)
            for _ in range(CAPTURE_COUNT):
                ret, f = cap.read()
                if not ret:
                    continue

                # 1) ROI(번호판 후보) 찾기
                roi = find_plate_roi(f)

                # OCR 결과 텍스트를 담을 변수
                text = ""

                # 2) ROI가 잡히면 ROI에 대해 OCR 우선 수행
                if roi is not None:
                    text = ocr_image(roi, OCR_KOR)

                # 3) ROI OCR 결과가 비어 있거나 의미 없으면 전체 프레임 OCR로 백업
                if not text.strip():
                    text = ocr_image(f, OCR_KOR)

                # 4) OCR 결과에서 번호판 형태로 파싱
                parsed = parse_plate(text)
                if parsed:
                    # 5) 허용 목록과 비교
                    allowed = match_plate(parsed, WHITELIST)

                    # 화면에 표시할 텍스트(여기선 last4 + confidence만 표시)
                    result_text = f"{parsed['last4']} ({parsed['confidence']})"

                    # LED/결과 상태 저장
                    result_allowed = allowed
                    result_time = now

                    # 결과 표시 상태로 전환 후, 스캔 루프 탈출
                    state = "RESULT"
                    break

    # ================= RESULT 상태 =================
    elif state == "RESULT":
        # 허용이면 초록, 아니면 빨강
        set_led(g=result_allowed, r=not result_allowed)

        # 결과 텍스트 표시
        cv2.putText(frame, result_text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0) if result_allowed else (0, 0, 255), 3)

        # RESULT_HOLD(3초) 지나면 다시 IDLE로 복귀
        if now - result_time > RESULT_HOLD:
            state = "IDLE"

    # 화면 출력 창
    cv2.imshow("Parking OCR Stable", frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료 시 자원 정리
cap.release()
cv2.destroyAllWindows()

# LED 모두 끄기
set_led(False, False, False)
