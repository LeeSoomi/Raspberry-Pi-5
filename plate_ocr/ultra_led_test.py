#사전 설치
sudo apt install -y python3-rpi.gpio

import RPi.GPIO as GPIO
import time

# ===== 핀 설정 (BCM) =====
TRIG = 17
ECHO = 27

LED_RED = 23
LED_GREEN = 12
LED_YELLOW = 20

# ===== 파라미터 =====
DIST_THRESHOLD_CM = 40.0   # 이 거리 이하이면 "접근"으로 판단
GREEN_SEC = 5
RED_SEC = 2

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.setup(LED_RED, GPIO.OUT)
GPIO.setup(LED_GREEN, GPIO.OUT)
GPIO.setup(LED_YELLOW, GPIO.OUT)

def set_led(red=False, green=False, yellow=False):
    GPIO.output(LED_RED, red)
    GPIO.output(LED_GREEN, green)
    GPIO.output(LED_YELLOW, yellow)

def get_distance_cm(timeout_sec=0.03):
    """
    HC-SR04 거리(cm) 측정.
    timeout_sec: 에코가 안 돌아올 때 무한 대기 방지
    """
    # TRIG 안정화
    GPIO.output(TRIG, False)
    time.sleep(0.0002)

    # 10us 펄스
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    t0 = time.time()
    # ECHO 상승 대기
    while GPIO.input(ECHO) == 0:
        if time.time() - t0 > timeout_sec:
            return None
    start = time.time()

    # ECHO 하강 대기
    while GPIO.input(ECHO) == 1:
        if time.time() - start > timeout_sec:
            return None
    end = time.time()

    # 거리 계산
    elapsed = end - start
    distance_cm = (elapsed * 34300) / 2.0
    return distance_cm

def main():
    print("초음파+LED 테스트 시작 (종료: Ctrl+C)")
    set_led(yellow=True)  # 기본 상태

    try:
        while True:
            d = get_distance_cm()

            if d is None:
                print("거리 측정 실패(타임아웃). 배선/전원/레벨시프터 확인 필요.")
                set_led(yellow=True)
                time.sleep(0.2)
                continue

            print(f"거리: {d:.1f} cm")

            if d <= DIST_THRESHOLD_CM:
                print("접근 감지! -> GREEN 5초, 이후 RED 2초")
                set_led(green=True)
                time.sleep(GREEN_SEC)
                set_led(red=True)
                time.sleep(RED_SEC)
                set_led(yellow=True)
                time.sleep(0.5)  # 재트리거 방지 약간의 딜레이
            else:
                set_led(yellow=True)
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n종료합니다.")

    finally:
        set_led(False, False, False)
        GPIO.cleanup()

if __name__ == "__main__":
    main()

#실행방법  python3 ultra_led_test.py
