# 1. 데이터
import numpy as np

# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18]) # pred = predict 

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim = 1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(7))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(500))
# model.add(Dense(500))
# model.add(Dense(500))
# model.add(Dense(500))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100000, batch_size = 3)

# 4. 평가와 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 1) 
print("loss: ", loss)
print("acc:", acc)

y_pred = model.predict(x_pred)
print("y_pred:", y_pred)