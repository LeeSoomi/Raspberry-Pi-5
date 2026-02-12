import RPi.GPIO as GPIO
import pygame
import time

pygame.init()

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 500

SCREEN = pygame.display.set_mode( (SCREEN_WIDTH, SCREEN_HEIGHT) )
pygame.display.set_caption("AI car")

t = 0.01

Motor_R1_Pin = 9
Motor_R2_Pin = 10
Motor_L1_Pin = 27
Motor_L2_Pin = 22
Enable_1 = 17
Enable_2 = 11

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(Motor_R1_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_R2_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_L1_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_L2_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Enable_1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Enable_2, GPIO.OUT, initial=GPIO.LOW)

pwm_e1 = GPIO.PWM(Enable_1, 100)
pwm_e2 = GPIO.PWM(Enable_2, 100)
pwm_e1.start(0)
pwm_e2.start(0)

def brake():
    pwm_e1.ChangeDutyCycle(10)
    pwm_e2.ChangeDutyCycle(10)
    GPIO.output(Motor_R1_Pin, False)
    GPIO.output(Motor_R2_Pin, True)
    GPIO.output(Motor_L1_Pin, False)
    GPIO.output(Motor_L2_Pin, True)
    time.sleep(t)
    GPIO.output(Motor_R1_Pin, False)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, False)
    GPIO.output(Motor_L2_Pin, False)
    print("Break")

def stop():
    pwm_e1.ChangeDutyCycle(100)
    pwm_e2.ChangeDutyCycle(100)
    GPIO.output(Motor_R1_Pin, False)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, False)
    GPIO.output(Motor_L2_Pin, False)
    print("Stop")

def forward_l():
    pwm_e1.ChangeDutyCycle(15)
    pwm_e2.ChangeDutyCycle(15)
    GPIO.output(Motor_R1_Pin, True)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, True)
    GPIO.output(Motor_L2_Pin, False)
    print("Forward_Low")
    time.sleep(t)

def forward_f():
    pwm_e1.ChangeDutyCycle(30)
    pwm_e2.ChangeDutyCycle(30)
    GPIO.output(Motor_R1_Pin, True)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, True)
    GPIO.output(Motor_L2_Pin, False)
    print("Forward_Fast")
    time.sleep(t) 

def Reverse():
    pwm_e1.ChangeDutyCycle(15)
    pwm_e2.ChangeDutyCycle(15)
    GPIO.output(Motor_R1_Pin, False)
    GPIO.output(Motor_R2_Pin, True)
    GPIO.output(Motor_L1_Pin, False)
    GPIO.output(Motor_L2_Pin, True)
    print("Reverse")
    time.sleep(t)

def turnLeft():
    pwm_e1.ChangeDutyCycle(15)
    pwm_e2.ChangeDutyCycle(20)
    GPIO.output(Motor_R1_Pin, True)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, False)
    GPIO.output(Motor_L2_Pin, True)
    print ("Turn_Left")
    time.sleep(t)

def turnLeft_f():
    pwm_e1.ChangeDutyCycle(20)
    pwm_e2.ChangeDutyCycle(20)
    GPIO.output(Motor_R1_Pin, True)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, False)
    GPIO.output(Motor_L2_Pin, True)
    print ("Turn_Left_Fast")
    time.sleep(t)

def turnRight():
    pwm_e1.ChangeDutyCycle(20)
    pwm_e2.ChangeDutyCycle(15)
    GPIO.output(Motor_R1_Pin, False)
    GPIO.output(Motor_R2_Pin, True)
    GPIO.output(Motor_L1_Pin, True)
    GPIO.output(Motor_L2_Pin, False)
    print("Turn_Right")
    time.sleep(t)

def turnRight_f():
    pwm_e1.ChangeDutyCycle(20)
    pwm_e2.ChangeDutyCycle(20)
    GPIO.output(Motor_R1_Pin, False)
    GPIO.output(Motor_R2_Pin, True)
    GPIO.output(Motor_L1_Pin, True)
    GPIO.output(Motor_L2_Pin, False)
    print("Turn_Right_Fast")
    time.sleep(t)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                turnLeft()
            if event.key == pygame.K_RIGHT:
                turnRight()
            if event.key == pygame.K_UP:
                forward_l()
            if event.key == pygame.K_DOWN:
                Reverse()
        if event.type == pygame.KEYUP:
            stop()
