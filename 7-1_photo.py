# 실습 7-1 · Picamera2로 사진 한 장 찍기 (라즈베리파이 5, CSI 카메라)
from picamera2 import Picamera2
import time

camera = Picamera2()
config = camera.create_still_configuration(
    main={"size": (640, 480)})
camera.configure(config)

camera.start()
time.sleep(2)              # 밝기를 맞출 시간
camera.capture_file("cos.jpg")
print("사진 저장 완료")

camera.stop()
camera.close()
