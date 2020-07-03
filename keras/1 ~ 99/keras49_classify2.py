# onehotencoding


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM
from keras.callbacks import EarlyStopping

# 1. 데이터

x= np.array(range(1,11))
y= np.array([1,2,3,4,5,1,2,3,4,5])


print(x.shape)
print(y.shape)
print("=" * 30)
y = y.reshape(-1, 1)
# print(y)
# print("=" * 30)
print(y.shape)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()
print(y.shape)
print(y)
# print("=" * 30)

# 2. 모델

input1 = Input(shape = (1,))
dense1 = Dense(100, activation = 'relu')(input1)
dense2 = Dense(50)(dense1)
dense3 = Dense(30)(dense2)
dense4 = Dense(15)(dense3)

output1 = Dense(5)(dense4)
output2 = Dense(5, activation= 'softmax')(output1)

model = Model(inputs = input1, outputs = output2)
# model = Sequential()
# model.add(Dense(10, activation = 'relu', input_dim = 1))
# model.add(Dense(20, activation = 'relu'))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(5, activation = 'relu'))
# model.add(Dense(1, activation = 'softmax'))

# 3. 훈련, 컴파일

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
model.fit(x, y, epochs = 100, batch_size = 1)

# 4. 평가 예측

loss, acc = model.evaluate(x, y, batch_size = 1)
print("loss:", loss)
print("mse:", acc)

x_pred = np.array([1,2,3,4,5])
y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred, axis = 1 ).reshape(-1,)
print(y_pred.shape)
print('=', y_pred)
