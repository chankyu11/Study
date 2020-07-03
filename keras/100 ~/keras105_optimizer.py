# 1. 데이터

import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

# 2. 모델

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD, Adagrad,Adadelta, Nadam

model = Sequential()
model.add(Dense(30,activation = 'relu', input_dim = 1))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))

# optimizer = Adam(lr = 0.001)
# optimizer = RMSprop(lr = 0.001)
# optimizer = SGD(lr = 0.001)
# optimizer = Adadelta(lr = 0.001)
# optimizer = Nadam(lr = 0.001)
optimizer = Adagrad(lr = 0.001)

model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse'])
model.fit(x, y, epochs = 100)

loss = model.evaluate(x,y)
print(loss)

pred1 = model.predict([3.5])
print(pred1)
'''
RMsprop
[0.0023732453119009733, 0.0023732453119009733]
[[3.4365735]]

Adam
[0.0382305383682251, 0.0382305383682251]
[[3.4120564]]

SGD
[0.01612633466720581, 0.01612633466720581]
[[3.4387863]]

Adadelta
[6.460569381713867, 6.460569381713867]
[[0.25141406]]

Nadam
[0.05996176600456238, 0.05996176600456238]
[[3.3884819]]

Adagrad
[5.574257850646973, 5.574257850646973]
[[0.45106423]]
'''