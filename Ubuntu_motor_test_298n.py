#!/usr/bin/env python3
"""
L298N 듀얼 모터 테스트 — 라즈베리파이 5 + Ubuntu
BCM 번호 (이미지 기준):
  왼쪽: ENB=17, IN4=27, IN3=22
  오른쪽: ENA=11, IN1=9, IN2=10
"""

#!/usr/bin/env python3
"""
L298N 듀얼 모터 이동 테스트 — 라즈베리파이 5 + Ubuntu
실행: python3 motor_test_l298n.py  (인자 없음)

BCM: 왼쪽 ENB=17, IN4=27, IN3=22 / 오른쪽 ENA=11, IN1=9, IN2=10
"""
from __future__ import annotations

import signal
import sys
import time

# 이동 테스트 설정 (필요 시 여기만 수정)
MOVE_SPEED = 0.35  # 0~1
MOVE_SEC = 2.0  # 전진·후진 각각 유지 시간(초)
PAUSE_SEC = 0.5  # 정지 대기(초)

try:
    from gpiozero import Device, DigitalOutputDevice, PWMOutputDevice
    from gpiozero.pins.lgpio import LGPIOFactory

    Device.pin_factory = LGPIOFactory()
except Exception:
    from gpiozero import DigitalOutputDevice, PWMOutputDevice


class L298NHalf:
    """L298N 한 채널: PWM(ENA/ENB) + 방향 2핀(IN)."""

    def __init__(self, pin_en: int, pin_a: int, pin_b: int) -> None:
        self._en = PWMOutputDevice(pin_en, frequency=1000)
        self._a = DigitalOutputDevice(pin_a)
        self._b = DigitalOutputDevice(pin_b)

    def set_speed(self, speed: float) -> None:
        s = max(-1.0, min(1.0, float(speed)))
        if abs(s) < 0.01:
            self._en.off()
            self._a.off()
            self._b.off()
            return
        if s > 0:
            self._a.on()
            self._b.off()
        else:
            self._a.off()
            self._b.on()
        self._en.value = abs(s)

    def stop(self) -> None:
        self.set_speed(0.0)

    def close(self) -> None:
        self.stop()
        self._en.close()
        self._a.close()
        self._b.close()


PIN_ENB, PIN_IN4, PIN_IN3 = 17, 27, 22
PIN_ENA, PIN_IN1, PIN_IN2 = 11, 9, 10


def run_movement_test(left: L298NHalf, right: L298NHalf) -> None:
    print(f"전진 ({MOVE_SPEED}, {MOVE_SEC}초)")
    left.set_speed(MOVE_SPEED)
    right.set_speed(MOVE_SPEED)
    time.sleep(MOVE_SEC)
    left.stop()
    right.stop()
    print("정지")
    time.sleep(PAUSE_SEC)

    print(f"후진 ({MOVE_SPEED}, {MOVE_SEC}초)")
    left.set_speed(-MOVE_SPEED)
    right.set_speed(-MOVE_SPEED)
    time.sleep(MOVE_SEC)
    left.stop()
    right.stop()
    print("정지 — 테스트 끝")


def main() -> int:
    left = L298NHalf(PIN_ENB, PIN_IN4, PIN_IN3)
    right = L298NHalf(PIN_ENA, PIN_IN1, PIN_IN2)

    def shutdown(*_a) -> None:
        left.stop()
        right.stop()
        left.close()
        right.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        run_movement_test(left, right)
    finally:
        left.close()
        right.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

