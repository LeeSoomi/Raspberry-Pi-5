# 라즈베리파이 5에서 GPIO를 사용하려면 RPi.GPIO 또는 최신 lgpio 라이브러리가 설치되어 있어야 합니다. 
# 최신 Raspberry Pi OS에는 기본적으로 포함되어 있다.

# 혹시 GPIO 관련 에러가 발생하면:
# bashsudo apt install python3-rpi.gpio
# # 또는
# sudo apt install python3-lgpio python3-rpi-lgpio   

from picamera2 import Picamera2
import cv2
import motor_module as motor
import RPi.GPIO as GPIO
import time
import numpy as np

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

Frame_Width  = 640
Frame_Height = 480

# Picamera2 초기화
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (Frame_Width, Frame_Height), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

time.sleep(2)  # 카메라 워밍업

try:
    while True:
        # 프레임 캡처
        frame = picam2.capture_array()
        
        # ✅ RGB를 BGR로 변환 (이 줄이 중요!)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = cv2.GaussianBlur(frame, (11, 11), 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
        upper_red = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        mask = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)
        
        mask = cv2.erode(mask, None, iterations=2)
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            try:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
                if radius > 5:
                    motor.forward_f()
                    if center[0] > Frame_Width/2 + 55:
                        motor.turnRight()
                    elif center[0] < Frame_Width/2 - 55:
                        motor.turnLeft()
                    else:
                        motor.forward_f()
                else:
                    motor.brake()
            except:
                pass
        else:
            motor.stop()
            
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
            
finally:
    motor.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
