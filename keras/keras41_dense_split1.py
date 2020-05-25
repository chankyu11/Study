import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM

# 1. 데이터

a = np.array(range(1,11))
size = 5    # time steps = 4

# lstm 모델 완성.

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("=========================================")
print(dataset)

print(dataset.shape)
print(type(dataset))

x = dataset[:,0:4]
# 0~4 즉 1,2,3,4 를 가져오겠다는 이야기
y = dataset[:,4]
#  4열을 즉 마지막 자리만 가져오겠다.
print(x)
print(y)

print(x.shape)
print(y.shape)

# x = x.reshape(x.shape[0],x.shape[1],1)
# x = np.reshape(x,(6,4,1)) 위와 같은 문법
print(x.shape)

# 2. 모델
model = Sequential()
model.add(Dense(10, activation = 'relu', input_dim = 4))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(25, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1))

model.summary()

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='loss', patience= 10, mode = 'auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=30, batch_size=1, verbose=1, callbacks = [es])

loss, mse = model.evaluate(x, y)

y_predict = model.predict(x)
print(y_predict)

print('loss', loss)
print('mse', mse)

