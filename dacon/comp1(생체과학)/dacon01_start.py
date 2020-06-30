import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
train = pd.read_csv('./dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./dacon/comp1/test.csv', header = 0, index_col = 0)
sm = pd.read_csv('./dacon/comp1/sample_submission.csv', header = 0, index_col = 0)

# print("train.shape: ", train.shape)   # 10000, 75 : x_train, test
# print("test.shape: ", test.shape)     # 10000, 71 : x_predict
# print("sm.shape: ", sm.shape)         # 10000, 4  : y_predict
# print(train.isnull().sum())

train = train.interpolate()  # 보간법 // 선형보간.
train = train.fillna(method = 'bfill')
print(train.isnull().sum())
# msno.matrix(train)
plt.show()            

# print(train)
# test = test.interpolate()
# test = test.fillna(method = 'bfill')


# # print(test)

# train_data = train.values
# test_data = test.values
# sample_data = sm.values

# np.save('./dacon/comp1/train_bfill.npy', arr = train_data)
# np.save('./dacon/comp1/test_bfill.npy', arr = test_data)
# np.save('./dacon/comp1/sample_submission_bfill.npy', arr = sample_data)


# # x_train = train.reshape[train.shape[-1], 71]

# # print(x_train.shape)