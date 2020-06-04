import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_boston

# # 1. 데이터

# ds = load_boston()

# x = ds.data
# y = ds.target

# # print(x.shape)  # 150, 4
# # print(y.shape)  # 150, 

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 256 ,train_size = 0.75)

# # 전처리
# scaler = StandardScaler()
# scaler.fit(x_train)
# x = scaler.transform(x_train)
# x1 = scaler.transform(x_test)

# # 2. 모델

# # model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=256)
# # model = LinearSVC()
# # model = SVC()
# # model = RandomForestClassifier()
# # model = RandomForestRegressor()
# # model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# # 3. 훈련

# model.fit(x_train, y_train)

# # 4. 평가
# score = model.score(x_test, y_test)
# pred = model.predict(x_test)

# # R2구하기
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, pred)

# print("score:", score)
# print("R2: ", r2)
# print('pred: ',pred)
'''
LinearSVC()
ValueError: Unknown label type: 'continuous'

SVC()
ValueError: Unknown label type: 'continuous'

RandomForestClassifier()
ValueError: Unknown label type: 'continuous'

RandomForestRegressor()
score: 0.9104612958191084
R2:  0.9104612958191084

KNeighborsClassifier()
ValueError: Unknown label type: 'continuous'

KNeighborsRegressor()
score: 0.5517513884461923
R2:  0.5517513884461923
'''

# # 분류

# 1. 데이터

ds = load_boston()

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
# model = SVC()
# model = RandomForestClassifier()
# model = RandomForestRegressor()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# 3. 훈련

model.fit(x_train, y_train)

# 4. 평가
score = model.score(x_test, y_test)
pred = model.predict(x_test)
print("score:", score)
print('정확도 :', accuracy_score(y_test, pred))

# '''
# LinearSVC()
# ValueError: Unknown label type: 'continuous'

# SVC()
# ValueError: Unknown label type: 'continuous'

# RandomForestClassifier()
# ValueError: continuous is not supported

# RandomForestRegressor()
# ValueError: continuous is not supported

# KNeighborsClassifier()
# ValueError: continuous is not supported

# KNeighborsRegressor()
# ValueError: continuous is not supported

# '''