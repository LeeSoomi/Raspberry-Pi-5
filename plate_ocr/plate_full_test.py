import cv2
import pytesseract
import re
from collections import Counter

# ===== Tesseract 경로 =====
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ===== 전체 번호판 정규식 (한국) =====
PLATE_REGEX = re.compile(
    r'([가-힣]{0,2}\d{2,3}[가-힣]\d{4})'
)

# ===== OCR 설정 =====
TESS_CONFIGS = [
    "--oem 3 --psm 7 -l kor+eng",
    "--oem 3 --psm 6 -l kor+eng"
]

# ===== 전체 번호판 추출 =====
def extract_full_plate(text):
    if not text:
        return None
    t = text.replace(" ", "")
    t = t.replace("|", "1").replace("I", "1").replace("O", "0")
    m = PLATE_REGEX.findall(t)
    return m[0] if m else None

# ===== OCR 실행 =====
def ocr_full_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = []

    for cfg in TESS_CONFIGS:
        raw = pytesseract.image_to_string(gray, config=cfg)
        plate = extract_full_plate(raw)
        if plate:
            results.append(plate)

    return results

# ===== 가장 많이 나온 번호 선택 =====
def vote_plate(plates):
    if not plates:
        return None
    return Counter(plates).most_common(1)[0][0]

# ===== 메인 =====
def main():
    cap = cv2.VideoCapture(0)

    # 해상도 높게 (중요)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("번호판 인식 테스트 시작 (종료: q)")

    buffer = []  # OCR 결과 누적

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 중앙 ROI (테스트용, 번호판을 가운데 두세요)
        h, w, _ = frame.shape
        roi = frame[int(h*0.4):int(h*0.6), int(w*0.2):int(w*0.8)]

        plates = ocr_full_plate(roi)
        buffer.extend(plates)

        # 10회 중 가장 많이 나온 번호 표시
        display_plate = vote_plate(buffer[-10:])

        if display_plate:
            cv2.putText(
                frame,
                f"PLATE: {display_plate}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

        # ROI 표시
        cv2.rectangle(
            frame,
            (int(w*0.2), int(h*0.4)),
            (int(w*0.8), int(h*0.6)),
            (255, 255, 0),
            2
        )

        cv2.imshow("Plate Full Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
