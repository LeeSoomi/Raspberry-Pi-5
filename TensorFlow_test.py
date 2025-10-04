# TensorFlow_test.py

import sys
print(f"Python 경로: {sys.executable}")
print(f"Python 버전: {sys.version}")
print()

try: 
  import tflite_runtime.interpreter as tflite
  print("✅ TensorFlow Lite 로드 성공!")
except ImportError as e: 
  print(f"❌ TensorFlow Lite 로드 실패: {e}") 
  print("\n가상환경이 활성화되었는지 확인하세요:") 
  print(" source ~/cos/env/bin/activate") sys.exit(1)
  
try: 
  import numpy as np
  print(f"✅ NumPy 버전: {np.__version__}")

except ImportError: print("❌ NumPy 로드 실패")
  print("\n 모든 라이브러리 정상 작동!")
