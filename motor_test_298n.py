#!/usr/bin/env python3
"""
L298N 듀얼 모터 테스트 — 라즈베리파이 5 + Ubuntu
BCM 번호 (이미지 기준):
  왼쪽: ENB=17, IN4=27, IN3=22
  오른쪽: ENA=11, IN1=9, IN2=10
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

# Pi 5: lgpio 백엔드 권장 (apt: python3-lgpio python3-gpiozero)
try:
    from gpiozero import Device, DigitalOutputDevice, PWMOutputDevice
    from gpiozero.pins.lgpio import LGPIOFactory

    Device.pin_factory = LGPIOFactory()
except Exception:
    from gpiozero import DigitalOutputDevice, PWMOutputDevice


class L298NHalf:
    """L298N 한 채널: PWM(ENA/ENB) + 방향 2핀(IN)."""

    def __init__(self, pin_en: int, pin_a: int, pin_b: int, name: str = "") -> None:
        self._name = name
        self._en = PWMOutputDevice(pin_en, frequency=1000)
        self._a = DigitalOutputDevice(pin_a)
        self._b = DigitalOutputDevice(pin_b)

    def set_speed(self, speed: float) -> None:
        """speed: -1.0 ~ 1.0 (음수면 반대 방향)."""
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


# 배선 (BCM)
PIN_ENB, PIN_IN4, PIN_IN3 = 17, 27, 22  # 왼쪽
PIN_ENA, PIN_IN1, PIN_IN2 = 11, 9, 10  # 오른쪽


def main() -> int:
    parser = argparse.ArgumentParser(description="L298N 모터 테스트 (BCM 핀 고정)")
    parser.add_argument("--left", type=float, default=None, help="왼쪽 모터 속도 -1~1")
    parser.add_argument("--right", type=float, default=None, help="오른쪽 모터 속도 -1~1")
    parser.add_argument("--both", type=float, default=None, help="양쪽 동일 속도 -1~1")
    parser.add_argument("--duration", type=float, default=2.0, help="동작 시간(초)")
    parser.add_argument("--sequence", action="store_true", help="순차 데모(전진/후진/정지)")
    args = parser.parse_args()

    left = L298NHalf(PIN_ENB, PIN_IN4, PIN_IN3, "left")
    right = L298NHalf(PIN_ENA, PIN_IN1, PIN_IN2, "right")

    def shutdown(*_a) -> None:
        left.stop()
        right.stop()
        left.close()
        right.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        if args.sequence:
            print("전진 0.4 …")
            left.set_speed(0.4)
            right.set_speed(0.4)
            time.sleep(args.duration)
            print("정지")
            left.stop()
            right.stop()
            time.sleep(0.5)
            print("후진 0.4 …")
            left.set_speed(-0.4)
            right.set_speed(-0.4)
            time.sleep(args.duration)
            print("정지")
            left.stop()
            right.stop()
        else:
            lv = args.both if args.both is not None else (args.left if args.left is not None else 0.0)
            rv = args.both if args.both is not None else (args.right if args.right is not None else 0.0)
            if args.left is None and args.right is None and args.both is None:
                parser.error("--left/--right/--both 중 하나 또는 --sequence 를 지정하세요.")
            print(f"왼쪽={lv}, 오른쪽={rv}, {args.duration}초")
            left.set_speed(lv)
            right.set_speed(rv)
            time.sleep(args.duration)
            left.stop()
            right.stop()
    finally:
        left.close()
        right.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
