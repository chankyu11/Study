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

# 1. 데이터

'''
# 데이터 저장
ds = pd.read_csv('./data/CSV/winequality-white.csv', 
                    index_col = 0, header = 0 , encoding = 'cp949', 
                    sep = ';')
print(ds.head(5))
# for i in range(len(ds.index)):
#     for j in range(len(ds.iloc[i])):
#         ds.iloc[i,j] = ds.iloc[i,j].replace(';', ', ')

np.save('./data/wine.npy', arr = ds)
'''

wq_data = np.load('./data/wine.npy', allow_pickle = True)

# print(wq_data)
# print(wq_data.shape)   # (4898, 12)

# 전처리

x = wq_data[:, :11]
y = wq_data[:, -1]
# print(x.shape)
# print(y.shape)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 256 ,train_size = 0.75)

scaler = StandardScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
x1 = scaler.transform(x_test)

pca = PCA(n_components= 10)
pca.fit(x_train)
x_train = pca.transform(x_train)
pca.fit(x_test)
x_test = pca.transform(x_test)

print(x_train.shape)
print(x_test.shape)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(y_test)
# print(y_train)
# print(y_test.shape)
# print(y_train.shape)

# # print(x_train.shape)  # (3673, 11)
# # print(x_test.shape)   # (1225, 11)
# # print(y_train.shape)  # (3673, )
# # print(y_test.shape)   # (1225, ) 

# 2. 모델

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape = (10, )))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(5))
model.add(Dense(10, activation = 'softmax'))


# 3. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 32)

# 4. 평가

acc = model.evaluate(x_test,y_test)
# score = model.score(x_test, y_test)
# pred = model.predict(x_test)

# y_pred = model.predict(x_test, y_test)
# print(y_pred)

# print("score:", score)
print('acc: ', acc)
# print('정확도 :', accuracy_score(y_test, y_pred))