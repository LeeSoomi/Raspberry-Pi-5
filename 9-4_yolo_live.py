# 확장 · YOLO로 카메라 속 사물 검출하기 (q키로 종료)
# 필요: pip install ultralytics, 처음 실행 때 yolov8n.pt 모델 다운로드(인터넷)
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2

model = YOLO("yolov8n.pt")          # 80가지 사물을 배운 작은 사전학습 모델
camera = Picamera2()
camera.configure(camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}))
camera.start()

while True:
    img = camera.capture_array()
    results = model(img, verbose=False)
    cv2.imshow("YOLO", results[0].plot())   # 네모와 이름을 그린 사진
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
