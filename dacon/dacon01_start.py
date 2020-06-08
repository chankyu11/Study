import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
sm = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)

print("train.shape: ", train.shape)   # 10000, 75 : x_train, test
print("test.shape: ", test.shape)     # 10000, 71 : x_predict
print("sm.shape: ", sm.shape)         # 10000, 4  : y_predict

print(train.isnull().sum())

train = train.interpolate()  # 보간법 // 선형보간.
train = train.fillna(0)
# print(train.isnull().sum())
print(train)
test = test.interpolate()
# print(test)

train_data = train.values
test_data = test.values
sample_data = sm.values

np.save('./data/dacon/comp1/train2.npy', arr = train_data)
np.save('./data/dacon/comp1/test2.npy', arr = test_data)
np.save('./data/dacon/comp1/sample_submission2.npy', arr = sample_data)


# x_train = train.reshape[train.shape[-1], 71]

# print(x_train.shape)