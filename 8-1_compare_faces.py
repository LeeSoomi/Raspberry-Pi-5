# 실습 8-1 · 두 사진의 얼굴 비교하기
# img 폴더에 기준 사진(person.jpg)과 비교할 사진(test.jpg)을 넣습니다.
# 아무 키나 누르면 종료됩니다.
import face_recognition
import cv2

# 1. 사진 불러오기 (load_image_file은 이미 RGB 순서 → 따로 변환하지 않음)
known = face_recognition.load_image_file("img/person.jpg")
test = face_recognition.load_image_file("img/test.jpg")

# 2. 얼굴 위치 찾기 → 128개 숫자로 바꾸기
known_locs = face_recognition.face_locations(known)
test_locs = face_recognition.face_locations(test)
known_encs = face_recognition.face_encodings(
    known, known_locs)
test_encs = face_recognition.face_encodings(
    test, test_locs)

# 3. 얼굴이 없으면 [0]을 꺼낼 때 오류가 나므로 먼저 확인
if not known_encs or not test_encs:
    print("얼굴을 찾지 못했습니다.")
    exit()

known_enc = known_encs[0]           # 첫 번째 얼굴
test_enc = test_encs[0]

# 4. 비교하기
matched = face_recognition.compare_faces(
    [known_enc], test_enc, tolerance=0.6)[0]
distance = face_recognition.face_distance(
    [known_enc], test_enc)[0]
print("동일 인물 판정:", matched)
print("얼굴 거리:", round(distance, 3))

# 5. 결과를 사진 위에 표시 (화면 표시 때만 RGB → BGR)
top, right, bottom, left = test_locs[0]
show = cv2.cvtColor(test, cv2.COLOR_RGB2BGR)
cv2.rectangle(show, (left, top), (right, bottom),
              (255, 0, 255), 2)
cv2.putText(show, f"{matched} {distance:.2f}", (50, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
cv2.imshow("test", show)
cv2.waitKey(0)
cv2.destroyAllWindows()
