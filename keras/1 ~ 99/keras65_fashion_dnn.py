# 과제2
# Sequential형으로 완성.
# 하단에 주석으로 acc와 loss결과 명시.
import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

# 1. 데이터, 전처리

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_test.shape)  # 10000, 28, 28
print(x_train.shape) # 60000, 28, 28 
print(y_test.shape)  # 10000, 
print(y_train.shape) # 60000, 

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # 60000, 10
print(y_test.shape)  # 10000, 10
# print(y_train)
# print(y_test)

x_test = x_test.reshape(10000, 784,).astype('float32') / 255
x_train = x_train.reshape(60000,784,).astype('float32') / 255

# 2.모델

model = Sequential()
model.add(Dense(500,input_shape =(784, )))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(200))
model.add(Dense(100))
# model.add(Dense(100))
model.add(Dense(10, activation='softmax'))


# 3. 훈련, 컴파일

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
model.fit(x_train, y_train, epochs = 15, batch_size = 32)

# 4. 결과

loss, acc = model.evaluate(x_test,y_test)
print('loss:',loss)
print('acc:',acc)

'''
1. loss: 0.37965968008041384, acc: 0.8737999796867371
2. loss: 0.4005156459569931, acc: 0.8690999746322632
3. loss: 0.38276131764650345, acc: 0.8682000041007996
'''
