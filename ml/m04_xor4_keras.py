from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVC

# 1. 데이터
# xor 데이터
x_data = np.array([[0,0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([0, 1, 1, 0])

print(x_data.shape)
print(y_data.shape)

# 2. 모델

model = Sequential()
model.add(Dense(100, input_dim = 2, activation = 'relu'))
model.add(Dense(500))
model.add(Dense(2560))
model.add(Dense(300))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# model = KNeighborsClassifier(n_neighbors = 1)
# 어떤 값을 넣어줄까 적어야함 근처 범위

# 3. 훈련

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(x_data, y_data, epochs = 100, batch_size = 1)

# 4. 평가 예측

loss, acc = model.evaluate(x_data,y_data)
x_test = np.array([[0,0], [1, 0], [0, 1], [1, 1]])
y_predict = model.predict(x_test)

# acc = accuracy_score([0, 1, 1, 0], y_predict)
# evaluate = score  이 다음 predict

print(x_test, "의 예측 결과는", y_predict)
print("acc: ", acc)