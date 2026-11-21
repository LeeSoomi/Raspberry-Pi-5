# 실습 9-2 · TensorFlow로 옷 사진 분류하기 (카메라 필요 없음)
# 처음 실행할 때는 인터넷으로 데이터를 내려받으므로 시간이 걸립니다.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ① 데이터 불러오기: 학습용 60,000장, 시험용 10,000장 (28×28 흑백)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover',
               'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)    # (60000, 28, 28)
print(test_images.shape)     # (10000, 28, 28)

# ② 전처리: 0~255 → 0~1 (정규화) 후 25장 미리 보기
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# ③ 모델 만들기 → 학습 방법 정하기 → 학습 → 평가
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # 784개를 한 줄로
    keras.layers.Dense(128, activation='relu'),     # 뉴런 128개
    keras.layers.Dense(10, activation='softmax')    # 10가지 확률
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(
    test_images, test_labels, verbose=2)
print("시험 정확도:", test_acc)

# ④ 시험 사진 한 장 예측
select = 4
pred = model.predict(test_images[select:select+1])[0]

plt.imshow(test_images[select], cmap=plt.cm.binary)
plt.show()

print(pred)                       # 10개의 확률
print(np.argmax(pred))            # 가장 큰 확률의 번호
print("예측:", class_names[np.argmax(pred)])
print("정답:", class_names[test_labels[select]])
