import cv2

# 카메라 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다")
    exit()

print("카메라 테스트 시작 (q를 눌러 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Camera Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
