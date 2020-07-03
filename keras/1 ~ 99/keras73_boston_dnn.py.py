import numpy as np
from sklearn.datasets import load_boston
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, StandardScaler
from sklearn.decomposition import PCA

boston = load_boston()
# print(boston.DESER)

x = boston.data
y = boston.target

# print(boston)
print(x.shape) # (506, 13)
print(y.shape) # (506, )

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
# print(x)

# pca = PCA(n_components = 2)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = False, train_size = 0.8)

print(x_train.shape) # (404, 13)
print(x_test.shape)  # (102, 13)
print(y_train.shape) # (404,)
print(y_test.shape)  # (102,)

# 2. 모델

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape = (13, )))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

# 3. 훈련, 컴파일
es = EarlyStopping(monitor='loss', patience= 10, mode = 'auto')
modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', mode = 'auto', save_best_only = True)
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 100, validation_split= 0.25)

#4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test)
print('loss', loss)
print('mse', mse)

y_predict = model.predict(x_test)
# print(y_predict)

# R2구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)

