import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
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


def process1():
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

    # K-최근접 이웃 회귀 알고리즘1
    knr = KNeighborsRegressor(n_neighbors=3)
    # 예측 모델 훈련
    knr.fit(train_input, train_target)

    # 길이 50에 대한 무게 예측. -> 1033으로 예측.
    print(knr.predict([[50]]))

    # 길이 50에 대한 인접 이웃 포인트 찾기
    distances, indexes = knr.kneighbors([[50]])

    plt.scatter(train_input, train_target)
    plt.scatter(train_input[indexes], train_target[indexes], marker='D')

    plt.scatter(50, 1033, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()


def process2():
    # 두 변수 모두 1차원 넘파이 배열
    print(perch_length.shape, perch_weight.shape)

    # 타겟이 1차원 넘파이 배열일 경우 결과 Input 넘파이 배열도 1차원으로 생성됨.
    train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
    print(train_input.shape, test_input.shape)

    # 사이킷 런은 타깃 데이터로 2차원 넘파이 배열을 받기에 두 Input 데이터를 reshape을 통해서 2차원 넘파이 배열로 변경
    # 첫 번 째 변수를 -1로 두면 나머지 원소 개수로 모두 채움. ex) np.array([1, 2, 3]).reshape(-1, 1) -> [[1], [2], [3]]
    # reshape의 두번째 매개변수는 차원 내 속성 갯수라고 보면 된다.
    # 구하려는 y는 농어의 무게이고 주어진 변수는 농어의 길이인 x이다.
    # 변수를 구하는 과정에서 우선 직선 방정식으로 보면 a * x + b = y이다.
    train_input = train_input.reshape(-1, 1)
    test_input = test_input.reshape(-1, 1)
    print(train_input.shape, test_input.shape)

    # 선형 회귀 알고리즘
    lr = LinearRegression()
    # 예측 모델 훈련
    lr.fit(train_input, train_target)

    # 선형 회귀 알고리즘으로 길이 50에 대한 무게 예측. -> 1241.8
    print(lr.predict([[50]]))
    # 농어 무게 = a x 농어 길이 + b
    # coef_ : 선의 기울기 a. 머신러닝에서는 계수 또는 가중치
    # intercept_ : y 절편인 b. x가 0일때의 y값.
    print(lr.coef_, lr.intercept_)

    # 훈련된 모델로 훈련 데이터 점수 확인 : 0.9398463339976041
    print(lr.score(train_input, train_target))
    # 훈련된 모델로 테스트 데이터 점수 확인 : 0.824750312331356
    print(lr.score(test_input, test_target))
    ## 훈련 점수가 더 높으나 훈련 점수도 낮고 테스트 점수와의 차이도 크기 때문에 과소적합.

    # 훈련 데이터 화면에 점으로 그리기
    plt.scatter(train_input, train_target)
    # 직선 그리기.
    # 첫번째 매개변수는 x 값. 15, 50 사이의 선을 그리기 위한 x축 값
    # 두번째 매개변수는 y 값. 첫번째 매개변수와 동일하게 두 수식 사이의 선을 그리기 위한 y축 값
    plt.plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_])

    plt.scatter(50, 1241.8, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()


def process3():
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

    # 훈련 셋트의 값들을 제곱한 np array와 기본 훈련 셋트로 다중 피처를 만든다.
    # [훈련 값의 제곱, 훈련 값]형태로 만들어진 배열이 생성.
    # 기존 예제는 1차원 함수로 구성됨. 직선 방정식은 a * x + b = y이기에 하나의 x값만 셋팅되었지만
    # 곡선을 사용하기 위해 2차 방정식을 사용해야 한다. a * x^2 + b * x + c = y라는 함수로 정의되기에
    # x^2, x 이 두가지 값을 하나의 속성으로 가진 배열을 사용해야 한다.
    train_poly = np.column_stack((train_input ** 2, train_input))
    test_poly = np.column_stack((test_input ** 2, test_input))

    print(train_poly.shape, test_poly.shape)

    # 선형 회귀 알고리즘
    lr = LinearRegression()
    # 예측 모델 훈련
    lr.fit(train_poly, train_target)

    # 2차 함수 형태의 데이터를 훈련했기에 예측하려는 데이터도 2차 함수 형태로 설정한다.
    # 예측 값(Y) : 1573.98423528
    predictedValue = lr.predict([[50 ** 2, 50]])
    print(predictedValue)

    # 농어 무게 = a * 농어 길이^2 + b * 농어 길이 + c
    # coef_ : 선의 기울기 [a, b]. 머신러닝에서는 계수 또는 가중치
    # intercept_ : y 절편인 c. x가 0일때의 y값.
    print(lr.coef_, lr.intercept_)

    # 15에서 49까지의 정수로 이루어진 1차원 np array
    point = np.arange(15, 50)
    # 훈련 데이터 화면에 점으로 그리기
    plt.scatter(train_input, train_target)

    # 곡선을 그리기 위해서 두 점의 포인트가 아닌 과정(함수)가 필요.
    # 아래처럼 x의 값 리스트와 x의 값 리스트를 변수로 한 함수를 정의해주면 Matplotlib 내부에서 x 값 리스트에 따른 y값 리스트를 찾아 선을 그림.
    plt.plot(point, lr.coef_[0] * point**2 + lr.coef_[1] * point + lr.intercept_)

    plt.scatter([50], [predictedValue], marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()

    # 훈련된 모델로 훈련 데이터 점수 확인 : 0.9706807451768623
    print(lr.score(train_poly, train_target))
    # 훈련된 모델로 테스트 데이터 점수 확인 : 0.9775935108325122
    print(lr.score(test_poly, test_target))
    ## 적절히 높은 훈련 점수가 나왔으나 테스트 점수가 조금 높게 나온 과대적합 상태이다.


if __name__ == '__main__':
    # process1()
    # process2()
    process3()
