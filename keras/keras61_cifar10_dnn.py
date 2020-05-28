import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

# 1. 데이터, 전처리

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) # 50000, 32, 32, 3
print(x_test.shape)  # 10000, 32, 32, 3
print(y_train.shape) # 50000, 1
print(y_test.shape)  # 10000, 1

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10)
print(y_test.shape)  # ()
print(y_train)
print(y_test)

# # 데이터 전처리 (정규화)

# x_test = x_test.reshape(10000, 3072,).astype('float32') / 255
# x_train = x_train.reshape(50000, 3072,).astype('float32') / 255
# print(x_train.shape)
# print(x_test.shape)

# # 2. 모델

# model = Sequential()
# model.add(Dense(500,input_shape =(3072, )))
# model.add(Dense(1500, activation='relu'))
# model.add(Dropout(0.4))

# model.add(Dense(200))
# model.add(Dense(100))
# # model.add(Dense(100))
# model.add(Dense(10, activation='softmax'))

# model.summary()

# # 3. 훈련, 평가

# model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
# model.fit(x_train,y_train, epochs = 10,batch_size=32)


# #4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=16) #x,y를 평가하여 loss와 acc에 반환하겠다.
# print("loss : ", loss)
# print("acc : ", acc)

# y_test = np.argmax(y_test, axis=1)
# print(y_test)