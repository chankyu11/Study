# CSV 파일을 보면 해더 부분 첫 행, 첫 열을 붙여줄지 뺄지 생각해야함.
# 150,4,setosa,versicolor,virginica iris 파일의 해더 부분.

import numpy as np
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import Conv2D , MaxPooling2D, Dense, Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
# 1. 데이터

iris_data_load = np.load('./data/iris_data.npy')
print(iris_data_load.shape)

x = iris_data_load[:, 0:4]
y = iris_data_load[:, 4]

print(x.shape) # (150, 4)
print(y.shape) # (150, )

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
# print(x)
print(x.shape)

y_data = np_utils.to_categorical(y)
print(y_data.shape) # (150, 3)
print(y_data) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y_data, shuffle = True, train_size = 0.8)

print(y_test.shape) # (150, 3)
print(y_train.shape) # (150, 3)
print(y_test) # (150, 3)

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape) # (120, 3)
# print(y_test.shape)  # (30, 3)

# 2. 모델

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape = (4, )))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(5))
model.add(Dense(3, activation = 'softmax'))

model.summary()

# 3. 훈련, 컴파일

# es = EarlyStopping(monitor='val_loss', patience= 20, mode = 'auto')

# modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
# mcp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
#                         mode = 'auto', save_best_only = True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs =100, batch_size= 64,
                validation_split = 0.2, verbose = 2)


#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
print('loss', loss)
print('acc', acc)
