# 실습 7-3 · 컬러 사진을 흑백으로, 흑백을 이진화로 바꾸기
# 7-1에서 찍은 cos.jpg가 같은 폴더에 있어야 합니다. 아무 키나 누르면 종료됩니다.
import cv2

img = cv2.imread("cos.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img.shape)     # (480, 640, 3)
print(gray.shape)    # (480, 640)

# 127보다 밝으면 255(흰색), 127 이하이면 0(검정)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("color", img)
cv2.imshow("gray", gray)
cv2.imshow("binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
