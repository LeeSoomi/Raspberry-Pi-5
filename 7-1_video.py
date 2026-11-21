# 실습 7-1 · Picamera2로 30초 동영상 녹화하기
# 재생: 터미널에서 vlc cos.h264
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import time

camera = Picamera2()
config = camera.create_video_configuration(
    main={"size": (640, 480)})
camera.configure(config)

encoder = H264Encoder(bitrate=10000000)
camera.start_recording(encoder, "cos.h264")
time.sleep(30)             # 30초 동안 녹화
camera.stop_recording()
camera.close()
print("녹화 완료")
