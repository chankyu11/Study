from keras.models import Sequential # keras 안에 models가 있음 그리고 그 모델 안에 Sequential이 존재
from keras.layers import Dense # keras 안에 위와 마찬가지로 layers 그리고 그 안에 Dense가 존재.
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim = 1, activation = 'relu')) # dim은 차원, add 추가한다 층을 쌓겠다.
model.add(Dense(3))
model.add(Dense(1, activation = 'relu'))

model.summary()

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_data = (x_train, y_train))
# fit은 Sequential 안에 있는 명령어 훈련을 시킨다. epohs 훈련 횟수, batch size 묶음
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)

print("loss:", loss)
print("acc:", acc)

output = model.predict(x_test)
print("결과물:\n", output)