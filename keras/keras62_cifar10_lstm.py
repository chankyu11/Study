import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

# 1. 데이터, 전처리

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train[0])
# print('y_train:', y_train[0]) # 6

print(x_test.shape) # 10000, 32, 32, 3
print(x_train.shape) # 50000, 32, 32, 3
print(y_test.shape)  # 10000, 1
print(y_train.shape) # 50000, 1
print(x_train[0].shape)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10)
print(y_test.shape)
print(y_train)
print(y_test)

# 데이터 전처리 (정규화)

x_test = x_test.reshape(10000, 1024, 3).astype('float32') / 255
x_train = x_train.reshape(50000, 1024, 3).astype('float32') / 255
print(x_train.shape)
print(x_test.shape)

# 2. 모델

input1 = Input(shape=(1024,3))
dense1 = LSTM(200, activation='relu', return_sequences= 'True')(input1)
dense2 = LSTM(100,activation='relu')(dense1)
dense3 = Dense(50,activation='relu')(dense2)

output1 = Dense(200)(dense3)
output2 = Dense(50)(output1)
output3 = Dense(10, activation='softmax')(output2)

model = Model(inputs = input1, outputs = output3)

model.summary()

# 3. 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 1,batch_size=32)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

y_test = np.argmax(y_test, axis=1)
print(y_test)