import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU
from sklearn.model_selection import train_test_split

lotto = pd.read_csv('./project/2020-6-15lotto_data.csv', sep = ",", index_col = 0)

print(lotto.max.counter())