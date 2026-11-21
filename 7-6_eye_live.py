# 확장활동 · 실시간으로 얼굴과 눈 찾기 (ESC 키로 종료)
import cv2
from picamera2 import Picamera2

HAAR_DIR = cv2.data.haarcascades if hasattr(cv2, "data") else "/usr/share/opencv4/haarcascades/"

face_cascade = cv2.CascadeClassifier(
    HAAR_DIR + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(
    HAAR_DIR + "haarcascade_eye.xml")

camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"})
camera.configure(config)
camera.start()

while True:
    img = camera.capture_array()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h),
                      (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]     # 얼굴 부분만 (세로 먼저, 가로 나중)
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.stop()
cv2.destroyAllWindows()
