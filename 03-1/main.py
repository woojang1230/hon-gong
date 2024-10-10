import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])


def process():
    plt.scatter(perch_length, perch_weight)
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()

    # 두 변수 모두 1차원 넘파이 배열
    print(perch_length.shape, perch_weight.shape)

    # 타겟이 1차원 넘파이 배열일 경우 결과 Input 넘파이 배열도 1차원으로 생성됨.
    train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
    print(train_input.shape, test_input.shape)

    # 사이킷 런은 타깃 데이터로 2차원 넘파이 배열을 받기에 두 Input 데이터를 reshape을 통해서 2차원 넘파이 배열로 변경
    # 첫 번 째 변수를 -1로 두면 나머지 원소 개수로 모두 채움. ex) np.array([1, 2, 3]).reshape(-1, 1) -> [[1], [2], [3]]
    # reshape의 두번째 매개변수는 차원 내 속성 갯수라고 보면 된다.
    train_input = train_input.reshape(-1, 1)
    test_input = test_input.reshape(-1, 1)
    print(train_input.shape, test_input.shape)

    # K-최근접 이웃 회귀 알고리즘
    knr = KNeighborsRegressor()
    # 예측 모델 훈련
    knr.fit(train_input, train_target)
    # 예측 모델 훈련 성능 평가
    score = knr.score(test_input, test_target)
    print(score)

    # 테스트 Input 데이터로 예측값 찾기.
    test_predict = knr.predict(test_input)
    # mean_absolute_error : 타깃과 예측값 사이의 절댓값 오차를 평균하여 반환.
    mae = mean_absolute_error(test_target, test_predict)
    print(mae)

    # 과대적합, 과소적합
    # 과대적합 : 훈련 점수가 테스트 점수보다 클 때 그 점수 차이가 클 경우.
    # 과소적합 : 테스트 점수가 훈련 점수보다 크거나 두 점수 모두 낮을 경우.

    # 현 시점에서는 테스트 점수(0.99)가 훈련 점수(0.96)보다 높음. 과소적합
    print(knr.score(train_input, train_target))
    print(knr.score(test_input, test_target))

    # 과소적합 해결을 위해 모델을 복잡하게 셋팅. 참조 이웃 수를 줄임(기본값 5)
    # 이웃 수를 줄이면 국지적 패턴에 민감해짐.
    knr.n_neighbors = 3
    # 다시 훈련 후 훈련 점수와 테스트 점수 비교.
    knr.fit(train_input, train_target)
    # 훈련 점수(0.98)보다 테스트 점수(0.97)가 적고 두 점수 차이도 작음. -> 적절한 상태.
    print(knr.score(train_input, train_target))
    print(knr.score(test_input, test_target))


if __name__ == '__main__':
    process()
