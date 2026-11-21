# 확장 · 진짜 CNN으로 옷 사진 분류하기 (9-2의 Dense 모델과 비교)
# CNN은 계산이 많아 학습 시간이 더 걸립니다. 처음에는 epochs를 줄여서 시작하세요.
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Reshape((28, 28, 1),
                         input_shape=(28, 28)),            # 색 채널 1개 추가
    keras.layers.Conv2D(32, (3, 3), activation='relu'),    # 3×3 필터 32개
    keras.layers.MaxPooling2D((2, 2)),                     # 크기 절반으로
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)

test_loss, test_acc = model.evaluate(
    test_images, test_labels, verbose=2)
print("CNN 시험 정확도:", test_acc)
