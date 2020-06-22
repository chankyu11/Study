import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.stats import itemfreq
from scipy.stats import mode
from keras.utils import np_utils

lotto = pd.read_csv('./project/2020-6-15lotto_data.csv', sep = ",", index_col = 0, header = None)

pred = lotto.iloc[910:, 0:6].values
print(pred.shape)
# lotto = lotto.iloc[:,0:6].values

# lotto = lotto[:, 0:6]
# # y = lotto[700: 910, 0:6]
# # pred = lotto[910:, 0:6]
# # print(x.shape)
# # print(y.shape)
# # print(pred.shape)

# def split_xy5(ds, time_steps, y_column):
#     x, y = list(), list()
#     for i in range(len(ds)):
#         x_end_number = i + time_steps 
#         y_end_number = x_end_number + y_column

#         if y_end_number > len(ds):
#             break
#         tmp_x = ds[i:x_end_number, :]
#         tmp_y = ds[x_end_number:y_end_number , 0]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# x, y = split_xy5(lotto, 10, 6)
# print(x.shape)
# print(y.shape)
# print(y[0, :])
# # print(y.shape)
# # print(y)
# # print(x[0,:6])
# # print(y[0,:6])
# # x = x.reshape(x.shape[0], -1, 1)
# # y = y.reshape(-1, 6, 1)
# # x = np_utils.to_categorical(x)
# # y = np_utils.to_categorical(y)
# # print(x.shape)
# # print(y.shape)


# # y_r = np.argmax(y, axis = 1).reshape(-1,6)
# # print(y_r[0,:6])

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 256 ,train_size = 0.8)

# # # print(y_train.shape)
# # # print(y_test.shape)
# # # y_train = y_train.reshape(-1, 1)
# # # y_test = y_test.reshape(-1, 1)

# # # y_train = np_utils.to_categorical(y_train)
# # # y_test = np_utils.to_categorical(y_test)
# # # print(y_train.shape)
# # # print(y_test.shape)

# # 2. 모델

# model = Sequential()
# model.add(LSTM(128, activation = 'relu',return_sequences = True ,input_shape = (10, 6)))
# # model.add(LSTM(64, activation = 'relu', input_shape = (10, 6)))
# model.add(LSTM(25, activation = 'relu'))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(6))

# # 3. 컴파일
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
# hist = model.fit(x_train, y_train, epochs =100, batch_size= 32,
#                 validation_split = 0.2, verbose = 2)


# # 4. 예측

# loss, acc = model.evaluate(x_test, y_test)
# print("loss:" , loss)
# print("acc: ", acc)


# y_pred = model.predict(x_test)
# # y_pred = np.argmax(y_pred, axis = 1)
# print(np.around(y_pred))
# # print(y_pred)