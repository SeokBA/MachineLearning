from keras.models import Sequential
from keras import *
model = Sequential() # 층을 추가할 준비

# 배치 경사 하강법(Batch Gradient Descent)
model.fit(X_train, y_train, batch_size=len(trainX))
# 확률적 경사 하강법(Stochastic Gradient Descent, SGD)
model.fit(X_train, y_train, batch_size=len(trainX))
# 미니 배치 경사 하강법(Mini-Batch Gradient Descent)
model.fit(X_train, y_train, batch_size=32) #32를 배치 크기로 하였을 경우
# 모멘텀(Momentum)
keras.optimizers.SGD(lr = 0.01, momentum= 0.9)
# 아다그라드(Adagrad)
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
# 알엠에스프롭(RMSprop)
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
# 아담(Adam)
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
