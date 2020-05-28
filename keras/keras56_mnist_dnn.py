import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, LSTM
from keras.models import Sequential , Model

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


#  데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)

# 데이터 전처리 2.정규화
x_train = x_train.reshape(60000,784,).astype('float32') / 255 # 255 검정 0 백색
x_test = x_test.reshape(10000, 784,).astype('float32') / 255
print(x_train.shape)
print(x_test.shape)

# 2. 모델

model = Sequential()
model.add(Dense(10,input_shape =(784, )))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dropout(0.4))
model.add(Dense(300))
model.add(Dropout(0.4))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 훈련, 평가

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 10,batch_size=32)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

y_test = np.argmax(y_test, axis=1)
print(y_test)