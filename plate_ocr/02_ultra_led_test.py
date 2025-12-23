from gpiozero import DistanceSensor, LED
import time

# ===== 핀 설정 (BCM) =====
ultrasonic = DistanceSensor(echo=27, trigger=17, max_distance=2.0)

led_red = LED(23)
led_green = LED(12)
led_yellow = LED(20)

DIST_THRESHOLD = 0.4  # meters (40cm)

def set_led(red=False, green=False, yellow=False):
    led_red.on() if red else led_red.off()
    led_green.on() if green else led_green.off()
    led_yellow.on() if yellow else led_yellow.off()

print("초음파 + LED 테스트 시작 (gpiozero)")
set_led(yellow=True)

try:
    while True:
        dist = ultrasonic.distance  # 0.0 ~ 1.0 (비율)
        if dist is None:
            continue

        cm = dist * 100
        print(f"거리: {cm:.1f} cm")

        if cm <= 40:
            print("접근 감지 → GREEN 5초 → RED 2초")
            set_led(green=True)
            time.sleep(5)
            set_led(red=True)
            time.sleep(2)
            set_led(yellow=True)
            time.sleep(0.5)
        else:
            set_led(yellow=True)
            time.sleep(0.2)

except KeyboardInterrupt:
    print("종료")

finally:
    set_led(False, False, False)
