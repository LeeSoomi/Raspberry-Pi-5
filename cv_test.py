import cv2
import numpy as np

# 빈 이미지 생성
img = np.zeros((400, 600, 3), dtype=np.uint8)
# 도형 그리기
cv2.rectangle(img, (50, 50), (550, 350), (0, 255, 0), 3)
cv2.circle(img, (300, 200), 80, (255, 0, 0), -1)
cv2.putText(img, "OpenCV Test", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
print("OpenCV 테스트 이미지 생성 완료")
print("창이 열립니다. 아무 키나 누르면 종료됩니다.")
cv2.imshow("OpenCV Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("OpenCV 정상 작동!")
