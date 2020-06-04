import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVC

# 회귀와, 분류로 만드시오.

# 회귀

# 1. 데이터

ds = load_iris()

x = ds.data
y = ds.target

# print(x.shape)  # 150, 4
# print(y.shape)  # 150, 

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 256 ,train_size = 0.75)

# 전처리
scaler = StandardScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
x1 = scaler.transform(x_test)

# 2. 모델

# model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=256)
# model = LinearSVC()
model = SVC()
# model = RandomForestClassifier()
# model = RandomForestRegressor()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# 3. 훈련

model.fit(x_train, y_train)

# 4. 평가
score = model.score(x_test, y_test)
pred = model.predict(x_test)

# R2구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, pred)

print("score:", score)
print("R2: ", r2)
print('pred: ',pred)

'''
RandomForestRegressor()
score: 0.9806215447154472
R2:  0.9806215447154472

RandomForestClassifier()
score: 1.0
R2:  1.0
pred:  [0 2 1 0 2 1 0 1 1 1 2 2 2 0 0 1 2 1 0 2 1 0 1 1 2 0 0 1 0 0 2 0 2 2 1 2 0
 0]

LinearSVC()
R2:  1.0
pred:  [0 2 1 0 2 1 0 1 1 1 2 2 2 0 0 1 2 1 0 2 1 0 1 1 2 0 0 1 0 0 2 0 2 2 1 2 0
 0]

SVC()
score: 0.9642857142857143
R2:  1.0
pred:  [0 2 1 0 2 1 0 1 1 1 2 2 2 0 0 1 2 1 0 2 1 0 1 1 2 0 0 1 0 0 2 0 2 2 1 2 0
 0]

KNeighborsClassifier()
score: 0.9553571428571429
R2:  0.9227642276422765
pred:  [0 2 2 0 2 1 0 1 1 1 2 2 1 0 0 1 2 1 0 2 1 0 1 1 2 0 0 1 0 0 2 0 2 2 1 2 0
 0]

KNeighborsRegressor()
score: 0.9534910671173347
R2:  0.9536585365853658
pred:  [0.  2.  1.6 0.  2.  1.  0.  1.  1.  1.2 1.8 2.  1.4 0.  0.  1.  1.8 1.
 0.  2.  1.  0.  1.  1.  2.  0.  0.  1.  0.  0.  2.  0.  1.6 1.6 1.  1.8
 0.  0. 
'''

# # 분류

# # 1. 데이터
# iris = load_iris()

# x1 = iris.data
# y1 = iris.target

# # 전처리
# scaler = StandardScaler()
# scaler.fit(x1)
# x1 = scaler.transform(x1)

# # 원핫인코딩
# from keras.utils.np_utils import to_categorical
# y1 = to_categorical(y1)
# # print(y.shape) # (150, 3)
# # print(y)

# x_train, x_test, y_train, y_test = train_test_split(x1, y1, random_state = 256 ,train_size = 0.8)

# # 2. 모델
# # model = SVC()
# # model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=256)
# # model = LinearSVC()
# # model = SVC()
# # model = RandomForestClassifier()
# # model = RandomForestRegressor()
# # model = KNeighborsClassifier()

# # 3. 훈련
# model.fit(x_train, y_train)

# # 4. 평가

# y_pred = model.predict(x_test)
# # print(y_pred)
# score = model.score(x_train, y_train)

# print("score:", score)
# print('정확도 :', accuracy_score(y_test, y_pred))

# '''
# RandomForestClassifier()
# score: 1.0
# 정확도 : 1.0

# RandomForestRegressor()
# error

# KNeighborsClassifier()
# score: 0.9583333333333334
# 정확도 : 0.9666666666666667

# KNeighborsRegressor()
# error

# LinearSVC()
# error (shape)

# SVC()
# error (shape)
# '''