import pandas as pd
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

# 와인 데이터 읽기

wine = pd.read_csv('./data/CSV/winequality-white.csv', sep = ';', header = 0)

y = wine['quality']
x = wine.drop('quality', axis = 1)
# y = wine 데이터의 퀄리티, x = 와인에서 quality를 버리겠다.

# y 레이블 축소

newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]

y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

y_pred = model.predict(x_test)

print("정답률: ", accuracy_score(y_test, y_pred))
print("acc: ", acc)