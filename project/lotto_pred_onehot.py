import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU
from sklearn.model_selection import train_test_split

lotto = pd.read_csv('./project/2020-6-15lotto_data.csv', sep = ",", index_col = 0, header= None)

lotto = lotto.iloc[:,0:6].values

# lotto = lo.values
print(lotto.shape)
print(type(lotto))
# print(lotto[0,0:])
y = lotto.reshape(-1,1)
# y = lotto.T

# # y = lotto
# # print(y.shape)

from keras.utils import np_utils
y = np_utils.to_categorical(y)
# y_test = np_utils.to_categorical(y_test)
print(y.shape)
print(y[0, 1:])

y_r = np.argmax(y, axis = 1).reshape(-1,6)
print(y_r.shape)
print(y_r[0,0:])
