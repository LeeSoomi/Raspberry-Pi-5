# 실습 9-1 · KNN으로 도미와 빙어 분류하기 (카메라 필요 없음)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# ① 데이터: 도미(bream) 35마리, 빙어(smelt) 14마리의 길이(cm)와 무게(g)
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.0, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0,
                475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0,
                575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0,
                920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0,
                14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.0, 9.8, 12.2, 13.4, 12.2,
                19.7, 19.9]

# ② 그래프 (기존 코드의 plt.colorbar()는 산점도에 필요 없어 삭제)
plt.scatter(bream_length, bream_weight, label="bream")
plt.scatter(smelt_length, smelt_weight, label="smelt")
plt.xlabel("length")
plt.ylabel("weight")
plt.legend()
plt.show()

# ③ 학습과 예측
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14     # 1 = 도미, 0 = 빙어

kn = KNeighborsClassifier()           # 이웃 5마리 확인
kn.fit(fish_data, fish_target)
print("학습 데이터 점수:", kn.score(fish_data, fish_target))
print("30cm, 600g 예측:", kn.predict([[30, 600]]))   # [1] → 도미

# ④ 보완: 학습용과 시험용 데이터를 나눠 공정하게 평가
train_input, test_input, train_target, test_target = \
    train_test_split(fish_data, fish_target,
                     test_size=0.25,
                     stratify=fish_target,
                     random_state=42)

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print("시험 점수:", kn.score(test_input, test_target))
