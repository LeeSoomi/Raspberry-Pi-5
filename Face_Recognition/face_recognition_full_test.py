#!/usr/bin/env python3
"""
Face Recognition + OpenCV 통합 테스트
"""
import sys

print("=" * 60)
print("Face Recognition 시스템 설치 확인")
print("=" * 60)

# 1. OpenCV 확인
try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
    cv2_path = cv2.__file__
    print(f"   위치: {cv2_path}")
except ImportError as e:
    print(f"❌ OpenCV: {e}")
    sys.exit(1)

# 2. NumPy 확인
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")
    sys.exit(1)

# 3. dlib 확인
try:
    import dlib
    print(f"✅ dlib: {dlib.__version__}")
    dlib_path = dlib.__file__
    print(f"   위치: {dlib_path}")
except ImportError as e:
    print(f"❌ dlib: {e}")
    print("   설치: pip3 install dlib --break-system-packages")
    sys.exit(1)

# 4. face_recognition 확인
try:
    import face_recognition
    print(f"✅ face_recognition: 설치 완료")
    fr_path = face_recognition.__file__
    print(f"   위치: {fr_path}")
except ImportError as e:
    print(f"❌ face_recognition: {e}")
    print("   설치: pip3 install face-recognition --break-system-packages")
    sys.exit(1)

print("=" * 60)

# 5. 기능 테스트
print("\n기능 테스트 중...")

# 더미 이미지 생성
test_image = np.zeros((100, 100, 3), dtype=np.uint8)
test_image[:] = (200, 200, 200)

# 얼굴 인식 시도 (얼굴 없는 이미지)
face_locations = face_recognition.face_locations(test_image)
print(f"✅ 얼굴 검출 기능: 정상 (검출된 얼굴: {len(face_locations)}개)")

print("=" * 60)
print("🎉 모든 시스템 정상 작동!")
print("=" * 60)

---------------------------
# #실행방법

# #실행 권한 부여
# bashchmod +x ~/code/face_recognition_full_test.py

# #실행
# # 방법 1
# python3 ~/code/face_recognition_full_test.py

# # 방법 2
# cd ~/code
# python3 face_recognition_full_test.py

# # 방법 3 (실행 권한 있으면)
# ~/code/face_recognition_full_test.py
