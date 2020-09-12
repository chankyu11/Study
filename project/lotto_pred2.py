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

# lo = lotto.iloc[:,0:6].values
# x = lotto.iloc[:457,0:6].values
# y = lotto.iloc[457:,0:6].values

x = lotto.iloc[:449,0:6].values
y = lotto.iloc[449:904, 0:6].values
pred = lotto.iloc[904:914, 0:6].values
# x = lotto[0:700, :]
# y = lotto[700: 914, :]
# pred = lotto[910:, :]
# print(x.shape)
# print(y.shape)
# print(pred.shape)
# print(lotto.shape)
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)
size = 10

"========================================================================"

def split_xy5(ds, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(ds)):
        x_end_number = i + time_steps 
        y_end_number = x_end_number + y_column

        if y_end_number > len(ds):
            break
        tmp_x = ds[i:x_end_number, :]
        tmp_y = ds[x_end_number:y_end_number , 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
# x, y = split_xy5(lotto, 10, 6)
x = split_x(x, size)
print(x.shape)   # 448, 10, 6
# print(y.shape)   # 459, 6
# print(y[0, :])

y = y[10:450, :] # 448, 6
print(y.shape)
print(pred.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 256 ,train_size = 0.8)

# 2. 모델

model = Sequential()
model.add(LSTM(128, activation = 'relu',return_sequences = True ,input_shape = (10, 6)))
# model.add(LSTM(64, activation = 'relu', input_shape = (10, 6)))
model.add(LSTM(25, activation = 'relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(6))

# 3. 컴파일
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs =100, batch_size= 32,
                validation_split = 0.2, verbose = 2)


# 4. 예측

loss, acc = model.evaluate(x_test, y_test)
print("loss:" , loss)
print("acc: ", acc)

pred = pred.reshape(1,10,6)
y_pred = model.predict(pred)
# y_pred = np.argmax(y_pred)
print(np.around(y_pred))
print(y_pred)