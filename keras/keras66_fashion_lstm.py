# Sequential형으로 완성.

import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input
from keras.utils import np_utils
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape) # 60000, 28, 28
print(x_test.shape)  # 10000, 28, 28
print(y_train.shape) # 60000,
print(y_test.shape)  # 10000,

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(60000,196,4).astype('float32') / 255
x_test = x_test.reshape(10000,196,4).astype('float32') / 255

# print(x_test.shape)  # 60000, 196, 4
# print(x_train.shape) # 60000, 196, 4
print(y_train.shape)
print(y_test.shape)

# 2. 모델

model = Sequential()
model.add(LSTM(10, input_shape = (196,4), activation = 'relu', return_sequences = 'True'))
model.add(LSTM(10))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(75))
model.add(Dense(30))
model.add(Dense(10, activation = 'softmax'))

# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10 ,batch_size = 32)

# 4.  평가, 예측

loss, acc = model.evaluate(x_test, y_test, batch_size = 32)

print('loss:', loss)
print('acc: ', acc)

'''
1. 
'''
