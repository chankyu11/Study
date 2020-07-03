from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt


# 1. 데이터, 전처리

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train[0])
print('y_train:', y_train[0]) # 6

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

x_test = x_test.astype('float32') / 255
x_train = x_train.astype('float32') / 255


# print(x_test)
# print()
# print("=" * 50)
# print()
# print(x_train)

# 2. 모델

model = Sequential()

model.add(Conv2D(50, (3,3), padding = 'same' ,input_shape = (32,32,3)))
model.add(Conv2D(150, (3,3), padding = 'same' ,input_shape = (32,32,3)))
model.add(Dropout(0.4))

model.add(Conv2D(100, (3,3), padding = 'same' ,input_shape = (32,32,3)))
model.add(MaxPooling2D(3,3))

model.add(Conv2D(80, (3,3), padding = 'same' ,input_shape = (32,32,3)))
model.add(MaxPooling2D(3,3))

model.add(Conv2D(50, (3,3), padding = 'same' ,input_shape = (32,32,3)))
model.add(Flatten())

model.add(Dense(100))
model.add(Dense(10, activation= 'softmax'))
model.summary()

# 3. 훈련, 컴파일

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
model.fit(x_train, y_train, epochs = 15, batch_size = 32)

# 4. 결과

result = model.evaluate(x_test,y_test)
print('result:',result)