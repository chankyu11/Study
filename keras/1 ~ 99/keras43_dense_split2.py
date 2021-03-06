# 42번 카피하여 Dense로 리뉴얼!
# 42번과 43번 비교! 더 좋은 값 찾기.


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM

# 1. 데이터

a = np.array(range(1,101))
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
print("===" * 10)
print(dataset)
print(dataset.shape)

# 마지막 6개의 행을 predict로 만들고 싶다.

predict = dataset[90:, :4]
print(predict)
print(predict.shape)

#  데이터 분할
x = dataset[:90, :4]
y = dataset[:90, [-1]]

print(x)
print(y)
print(x.shape)
print(y.shape)


# x = x.reshape(x.shape[0],x.shape[1],1)
# print(x.shape)

from sklearn.model_selection import train_test_split

# train, test 분리할 것.(8:2)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = False, train_size = 0.8)

print(x_train)
print(x_test)
print("=" * 40)
print(x_train.shape)
print(x_test.shape)

# 2. 모델
model = Sequential()
model.add(Dense(20, activation = 'relu', input_dim = 4))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

# input1 = Input(shape = (4,1))
# dense1 = LSTM(10, activation= 'relu',return_sequences = True)(input1)
# dense2 = LSTM(10, activation= 'relu')(dense1)
# dense3 = Dense(5,activation='relu')(dense2)

# output1 = Dense(10)(dense3)
# output2 = Dense(5)(output1)
# output3 = Dense(1)(output2)

# model = Model(inputs = input1, outputs = output3)

#3. 훈련

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='loss', patience= 10, mode = 'auto')

# validation을 넣을 것.(train의 20%)

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split= 0.25 ,verbose=1, callbacks = [es])

#4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test)
print('loss', loss)
print('mse', mse)

y_predict = model.predict(predict, batch_size=1)
print(y_predict)



