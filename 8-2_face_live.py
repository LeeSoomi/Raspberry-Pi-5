# 실습 8-2 · 여러 사람의 얼굴 실시간 인식 (q키로 종료)
# images 폴더에 등록할 사람의 사진을 넣습니다. 파일 이름이 화면에 보일 이름입니다.
# 예: images/minsu.jpg → MINSU
# 얼굴 사진은 본인 동의를 받은 것만 쓰고, 수업이 끝나면 삭제하세요.
import face_recognition
import cv2
import numpy as np
import os
from picamera2 import Picamera2

# ① 등록 사진 준비
path = "images"
known_encs = []          # 등록된 얼굴 숫자들
class_names = []         # 등록된 이름들

for file in os.listdir(path):
    img = face_recognition.load_image_file(f"{path}/{file}")
    encs = face_recognition.face_encodings(img)
    if encs:                       # 얼굴 있을 때만 등록
        known_encs.append(encs[0])
        class_names.append(os.path.splitext(file)[0])
print("등록된 사람:", class_names)

# ② 카메라 켜기
camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"})
camera.configure(config)
camera.start()

while True:
    img = camera.capture_array()              # BGR 순서
    small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locs = face_recognition.face_locations(small)
    encs = face_recognition.face_encodings(small, locs)

    # ③ 비교하고 이름 붙이기
    for enc, loc in zip(encs, locs):
        name = "Unknown"
        if known_encs:
            dist = face_recognition.face_distance(
                known_encs, enc)
            best = np.argmin(dist)
            if dist[best] <= 0.6:
                name = class_names[best].upper()

        top, right, bottom, left = [v * 4 for v in loc]   # 1/4로 줄였으니 4배
        cv2.rectangle(img, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        cv2.putText(img, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 2)

    # ④ 화면 표시와 종료
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
