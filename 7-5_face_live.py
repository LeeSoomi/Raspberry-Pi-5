# 실습 7-5 · 실시간 얼굴 검출 (ESC 키로 종료)
import cv2
from picamera2 import Picamera2

HAAR_DIR = cv2.data.haarcascades if hasattr(cv2, "data") else "/usr/share/opencv4/haarcascades/"

# 1. 얼굴 검출기 불러오기 (반복문 밖에서 한 번만)
face_cascade = cv2.CascadeClassifier(
    HAAR_DIR + "haarcascade_frontalface_default.xml")

# 2. 카메라 설정하고 켜기
camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"})
camera.configure(config)
camera.start()

# 3. 반복: 찍고 → 찾고 → 그리기
while True:
    img = camera.capture_array()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2,
        minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h),
                      (255, 0, 0), 2)

    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF == 27:    # 27 = ESC 키
        break

camera.stop()
cv2.destroyAllWindows()
