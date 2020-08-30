import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU, Conv1D, Input, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.stats import itemfreq
from scipy.stats import mode
from keras.utils import np_utils

lotto = pd.read_csv('D:/STUDY/project/2020-07-09-lotto_data.csv', sep = ",", index_col = 0, header = None)
# print(lotto.iloc[:,:6])    # (918, 6)

lt = lotto.iloc[:, :6].values

def split_xy5(ds, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(ds)):
        x_end_number = i + time_steps 
        y_end_number = x_end_number + y_column

        if y_end_number > len(ds):
            break
        tmp_x = ds[i:x_end_number, :]
        tmp_y = ds[x_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(lt, 10, 6)
# print(x.shape)     # (903, 10, 6)
# print(y.shape)     # (903, 6)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8 ,shuffle = False)

print(x_train.shape)    
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
''' 
(722, 10, 6)
(181, 10, 6)
(722, 6)
(181, 6) 
# '''

# 2. 모델

model = Sequential()
model.add(LSTM(64, activation = 'relu',return_sequences = True, input_shape = (10,6)))
model.add(LSTM(32, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(6, activation = 'softmax'))

model.summary()

# 3. 훈련

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 1, batch_size=32, validation_split= 0.2)

#4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test)
print('loss', loss)
# # print('mse', mse)
# # pred = pred.reshape(1,10,6)
# y_predict = model.predict(x_test)
# print(np.around(y_predict))

