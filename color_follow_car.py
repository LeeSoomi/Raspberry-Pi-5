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
        # RGB를 BGR로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = cv2.GaussianBlur(frame, (11, 11), 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
        upper_red = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        mask = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)
        
        mask = cv2.erode(mask, None, iterations=2)  # Do erode if needed
        # mask = cv2.dilate(mask, None, iterations=2)  # Do dilate if needed
        
        # Find the contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)  # Find the max length of contours
            ((x, y), radius) = cv2.minEnclosingCircle(c)  # Find the x, y, radius of given contours
            M = cv2.moments(c)  # Find the moments

            try:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # mass center
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)  # process every frame
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
                # Forward, backward, Turn rules
                # Size of the recognized object           
                if radius > 5:
                    motor.forward_f()
                    if center[0] > Frame_Width/2 + 55:
                        motor.turnRight()
                    elif center[0] < Frame_Width/2 - 55:
                        motor.turnLeft()
                    else:
                        motor.forward_f()  # fast Run
                else:
                    motor.brake()
            except:
                pass
        else:
            motor.stop()
            
        cv2.imshow("Frame", frame)  # if you don't need to display and the car will get faster
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
            
finally:  # except KeyboardInterrupt:
    motor.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()

============================
기존에 파일에서 카메라 관련 수정

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
        # RGB를 BGR로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = cv2.GaussianBlur(frame, (11, 11), 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
        upper_red = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        mask = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)
        
        mask = cv2.erode(mask, None, iterations=2)  # Do erode if needed
        # mask = cv2.dilate(mask, None, iterations=2)  # Do dilate if needed
        
        # Find the contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)  # Find the max length of contours
            ((x, y), radius) = cv2.minEnclosingCircle(c)  # Find the x, y, radius of given contours
            M = cv2.moments(c)  # Find the moments

            try:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # mass center
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)  # process every frame
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
                # Forward, backward, Turn rules
                # Size of the recognized object           
                if radius > 5:
                    motor.forward_f()
                    if center[0] > Frame_Width/2 + 55:
                        motor.turnRight()
                    elif center[0] < Frame_Width/2 - 55:
                        motor.turnLeft()
                    else:
                        motor.forward_f()  # fast Run
                else:
                    motor.brake()
            except:
                pass
        else:
            motor.stop()
            
        cv2.imshow("Frame", frame)  # if you don't need to display and the car will get faster
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
            
finally:  # except KeyboardInterrupt:
    motor.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()

# ===================================
# 수정된 내용
# # 기존
#    camera = cv2.VideoCapture(0)
#    camera.set(cv2.CAP_PROP_FRAME_WIDTH, Frame_Width)
#    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Frame_Height)
   
# # 변경
#    picam2 = Picamera2()
#    config = picam2.create_preview_configuration(
#        main={"size": (Frame_Width, Frame_Height), "format": "RGB888"}
#    )
#    picam2.configure(config)
#    picam2.start()
#    time.sleep(2)  # 카메라 워밍업
# -------------------
# # 기존
#    (_, frame) = camera.read()
   
#    # 변경
#    frame = picam2.capture_array()
#    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
