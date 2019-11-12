from tensorflow.keras.models import Sequential  # 케라스의 Sequential()을 임포트
from tensorflow.keras.layers import Dense  # 케라스의 Dense()를 임포트
from tensorflow.keras import optimizers  # 케라스의 옵티마이저를 임포트
import numpy as np  # Numpy를 임포트
import matplotlib.pyplot as plt # 맷플롯립 사용

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 공부하는 시간
y = np.array([11, 22, 33, 44, 53, 66, 77, 87, 95])  # 각 공부하는 시간에 맵핑되는 성적

model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))
sgd = optimizers.SGD(lr=0.01)
# 학습률(learning rate, lr)은 0.01로 합니다.
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
# sgd는 경사 하강법을 의미.
# 손실 함수(Loss function)은 평균제곱오차 mse를 사용합니다.
model.fit(X, y, batch_size=1, epochs=300, shuffle=False)
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 300번 시도합니다.

plt.plot(X, model.predict(X), 'b', X, y, 'k.')
plt.show()
print(model.predict([9.5]))
