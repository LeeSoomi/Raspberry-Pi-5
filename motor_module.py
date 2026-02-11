# RPi.GPIO 호환 레이어 설치
# sudo apt update
# sudo apt install python3-rpi-lgpio -y

from gpiozero import Motor, OutputDevice
import time

t = 0.01

# 모터 정의 (Motor 클래스 사용)
motor_left = Motor(forward=27, backward=22, enable=17)
motor_right = Motor(forward=9, backward=10, enable=11)

def brake():
    motor_left.forward(0.1)
    motor_right.forward(0.1)
    time.sleep(t)
    motor_left.stop()
    motor_right.stop()

def stop():
    motor_left.stop()
    motor_right.stop()

def forward_l():
    motor_left.forward(0.15)
    motor_right.forward(0.15)
    time.sleep(t)

def forward_f():
    motor_left.forward(0.3)
    motor_right.forward(0.3)
    time.sleep(t)

def Reverse():
    motor_left.backward(0.2)
    motor_right.backward(0.2)
    time.sleep(t)

def turnLeft():
    motor_left.backward(0)
    motor_right.forward(0.2)
    time.sleep(t)

def turnLeft_f():
    motor_left.backward(0.2)
    motor_right.forward(0.2)
    time.sleep(t)

def turnRight():
    motor_left.forward(0.2)
    motor_right.backward(0)
    time.sleep(t)

def turnRight_f():
    motor_left.forward(0.2)
    motor_right.backward(0.2)
    time.sleep(t)

def cleanup():
    motor_left.close()
    motor_right.close()
