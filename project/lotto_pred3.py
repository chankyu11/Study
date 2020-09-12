import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU, Conv1D, Input, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.stats import itemfreq
from scipy.stats import mode
from keras.utils import np_utils

lotto = pd.read_csv('D:/STUDY/project/2020-6-15lotto_data.csv', sep = ",", index_col = 0, header = None)

lt = lotto.iloc[:904, 0:6].values
# x = lotto.iloc[:449,0:6].values
# y = lotto.iloc[449:904, 0:6].values
pred = lotto.iloc[904:914, 0:6].values
# print(lt.shape)         # (904, 6)
# print(pred.shape)       # (10, 6)

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
print(x.shape)   # (889, 10, 6)
print(y.shape)   # (889, 6)
# print("===" * 50)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle = False, train_size = 0.8)

# 2. 모델

input1 = Input(shape = (10,6))
# dense1 = LSTM(10, activation= 'relu',return_sequences = True)(input1)
dense2 = Conv1D(64, 6,activation= 'relu')(input1)
dense2 = Conv1D(32, 1,activation= 'relu')(dense2)
dense2 = MaxPooling1D()(dense2)
dense2 = Flatten()(dense2)
dense3 = Dense(16,activation='relu')(dense2)

output1 = Dense(16)(dense3)
output2 = Dense(6)(output1)

model = Model(inputs = input1, outputs = output2)

model.summary()

# 3. 훈련

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=6, validation_split= 0.2 ,verbose=1)

#4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test)
print('loss', loss)
print('mse', mse)

# pred = pred.reshape(1,10,6)
y_predict = model.predict(pred)
print(np.around(y_predict))

