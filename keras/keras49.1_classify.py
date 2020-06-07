import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Activation
from keras.callbacks import EarlyStopping

# 1. 데이터

x= np.array(range(1,11))
y= np.array([1,2,3,4,5,1,2,3,4,5])

from keras.utils import np_utils
y = np_utils.to_categorical(y)
# categorical은 시작이 0부터 시작이라  1 = 0 1 0 0 0 0
print(y)
print(y.shape)

y = y[:,1:]
print(y.shape)

# 2. 모델

model = Sequential()
model.add(Dense(10, activation = 'relu', input_dim = 1))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))
model.add(Activation('sigmoid'))

model.summary()

# 3. 훈련, 컴파일

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
# 다중분류는 categorical_crossentropy activation = softmax
model.fit(x, y, epochs = 100, batch_size = 1)


# 4. 평가 예측

loss, acc = model.evaluate(x, y, batch_size = 1)
print("loss:", loss)
print("mse:", acc)

x_pred = np.array([1,2,3,4,5])
y_pred = model.predict(x_pred)

print('y_pred:', y_pred)
print(y_pred.shape)


'''
onehotencoding

1,2,3,4,5 라는 데이터를 인코딩하면

1,0,0,0,0
0,1,0,0,0
0,0,1,0,0
0,0,0,1,0
0,0,0,0,1 

이렇게 만듬
'''