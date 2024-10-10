import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 기본 특성 데이터
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

def process():
    # 특성 데이터들 집합의 리스트로 변경.
    fish_data = np.column_stack((fish_length, fish_weight))
    # 타깃 데이터 연결.
    fish_target = np.concatenate((np.ones(35), np.zeros(14)))

    print(fish_data.shape, fish_target.shape)

    # 입력 데이터 셋과 타깃 데이터 셋으로 학습, 테스트용 Input Data Set, Targer Data Set으로 자동 분류.
    # train_input, test_input -> 학습, 테스트용 입력 데이터셋
    # train_target, test_target -> 학습, 테스트용 타깃 데이터셋
    train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42, stratify=fish_target)

    print(train_input.shape, test_input.shape, train_target.shape, test_target.shape)

    target = [25, 150]

    # 평균 구하기
    mean = np.mean(train_input, axis=0)
    # 표준편차 구하기
    std = np.std(train_input, axis=0)

    # 표준점수로 변환
    # 학습 입력 데이터
    train_scaled = (train_input - mean) / std
    # 테스트 입력 데이터
    test_scaled = (test_input - mean) / std
    # 대상 데이터
    new_target = (target - mean) / std

    # 표준 점수로 변환된 데이터로 학습 및 검증 진행.
    kn = KNeighborsClassifier()
    # 학습 진행 - 학습 데이터로 진행
    kn.fit(train_scaled, train_target)
    # 검증 진행 - 테스트 데이터로 진행
    kn.score(test_scaled, test_target)

    # 대상 결과 확인 -> new_target은 1로 예측되어야 함.
    scoreResult = kn.predict([new_target])
    print(scoreResult)

    distances, indexes = kn.kneighbors([new_target])

    plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
    plt.scatter(new_target[0], new_target[1], marker='^')
    plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
    # plt.xlim((0, 1000))
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()


if __name__ == '__main__':
    process()
