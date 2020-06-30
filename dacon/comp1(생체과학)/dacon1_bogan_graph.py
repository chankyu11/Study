import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv('./dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./dacon/comp1/test.csv', header = 0, index_col = 0)
sm = pd.read_csv('./dacon/comp1/sample_submission.csv', header = 0, index_col = 0)

import missingno as msno

# 결측치 시각화  # train
# msno.matrix(train)
# plt.show()            

# 결측치 시각화  # test
# msno.matrix(test)
# plt.show()

a = train.isnull().sum()[train.isnull().sum().values > 0]
# print(a)

b = test.isnull().sum()[test.isnull().sum().values > 0]
# print(b)

# print(train.isnull().sum()[train.isnull().sum().values > 0].index)
# print(test.isnull().sum()[test.isnull().sum().values > 0].index)

# 광원 스펙트럼
# test.filter(regex='_src$',axis=1).head().T.plot()
# 측정 스펙트럼
test.filter(regex='_dst$',axis=1).head().T.plot()
# plt.show()

print(train.head())