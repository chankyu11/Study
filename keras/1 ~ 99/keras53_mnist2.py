import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
# print("=" * 20)
# print('y_train: ', y_train[0])
# print(y_train.shape)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(x_train[0].shape)
# plt.imshow(x_train[59999], 'gray')
# plt.imshow(x_train[0])
# plt.show()

# 데이터 전처리, 원핫인코딩

# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# enc.fit(y_train)
# enc.fit(y_test)
# y_test = enc.transform(y_test).toarray
# y_train = enc.transform(y_train).toarray

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10)
print(y_test.shape)
print(y_train)
print(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
# x_train의 shape를 4차원을 만들고, 그걸 float 타입으로 변환하고 거기에 除255
# minmax 는 0 ~ 1로 나오기에 값을 실수로 만들어야함.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# # 2. 모델
# from keras.models import Model, Sequential
# from keras.layers import Conv2D , MaxPooling2D, Dense, Flatten
# from keras.layers import Dropout

# model = Sequential()
# model.add(Conv2D(50, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# # model.add(MaxPooling2D(3,3))
# model.add(Conv2D(150, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# model.add(Conv2D(100, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(80, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(30, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Dense(10, activation= 'softmax'))
# model.summary()

# # 3. 훈련, 컴파일

# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
# model.fit(x_train, y_train, epochs = 15, batch_size = 32)


# # 4. 평가 예측

# result = model.evaluate(x_test,y_test)
# print('result:',result)

