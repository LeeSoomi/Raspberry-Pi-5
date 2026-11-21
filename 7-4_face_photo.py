# 실습 7-4 · 사진에서 얼굴을 찾아 네모 그리기
import cv2

# XML 폴더 위치 (apt로 설치한 OpenCV처럼 cv2.data가 없으면 직접 경로 사용)
HAAR_DIR = cv2.data.haarcascades if hasattr(cv2, "data") else "/usr/share/opencv4/haarcascades/"

face_cascade = cv2.CascadeClassifier(
    HAAR_DIR + "haarcascade_frontalface_default.xml")

img = cv2.imread("cos.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)   # scaleFactor, minNeighbors

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h),
                  (255, 0, 0), 2)                      # BGR 파란색, 두께 2

print("찾은 얼굴 수:", len(faces))
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
