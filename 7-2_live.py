# 실습 7-2 · OpenCV 창에 카메라 영상 띄우기 (q키로 종료)
# USB 웹캠을 쓴다면 기존 cv2.VideoCapture(0) 코드를 그대로 써도 됩니다.
import cv2
from picamera2 import Picamera2

camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"})   # OpenCV와 같은 BGR 순서
camera.configure(config)
camera.start()

while True:
    img = camera.capture_array()   # 사진 1장 = 숫자 배열
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
