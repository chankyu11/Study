import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

train = np.load('./dacon/comp1/train_bfill.npy')
test = np.load('./dacon/comp1/test_bfill.npy')
sample = np.load('./dacon/comp1/sample_submission_bfill.npy')

# print(train.shape)    # 10000, 75
# print(test.shape)     # 10000, 71
# print(sample.shape)   # 10000, 4

x = train[:, :71]       # 10000, 71
y = train[:, 71:]       # 10000, 4
# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 255)

# print(x_train.shape)    # 7500, 71
# print(x_test.shape)     # 2500, 71
# print(y_train.shape)    # 7500, 4
# print(y_test.shape)     # 2500, 4

scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.fit(x_test)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
test = scaler.fit_transform(test)

# 2. 모델

model = Sequential()
model.add(Dense(256, activation = 'relu', input_shape = (71, )))
model.add(Dense(128))
model.add(Dense(64))
# model.add(Dropout(0.4))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))

# model.summary()

# 3. 컴파일

model.compile(loss = 'mse', optimizer = 'adam', metrics= ['mae'])
model.fit(x_train_scaled, y_train, epochs = 100, validation_split = 0.15)

# 4. 평가

mae = model.evaluate(x_test_scaled, y_test)
y_pred = model.predict(test)
print('mae: ', mae)
print(y_pred)

a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred, a)
y_pred.to_csv('./dacon/comp1/sample_submission.csv', 
        index = True, header=['hhb','hbo2','ca','na'],index_label='id')