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
from sklearn.datasets import load_breast_cancer

# 1. 데이터

ds = load_breast_cancer()

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
model = KNeighborsClassifier()
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
LinearSVC()
score: 0.8671328671328671
R2:  0.42092924126172204

model = SVC()
score: 0.9230769230769231
R2:  0.6647485080988917

RandomForestClassifier()
score: 0.958041958041958
R2:  0.8171355498721227

RandomForestRegressor()
score: 0.8067945652173912
R2:  0.8067945652173912

KNeighborsClassifier()
score: 0.9440559440559441
R2:  0.7561807331628303

KNeighborsRegressor()
score: 0.8110400682011935
R2:  0.8110400682011935
'''