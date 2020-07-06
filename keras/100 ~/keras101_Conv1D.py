import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Flatten

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

pred = dataset[90:, :4]
print(pred)
print(pred.shape)

#  데이터 분할
x = dataset[:90, :4]
y = dataset[:90, [-1]]

print(x)
print(y)
print(x.shape)  # 90, 4
print(y.shape)  # 90, 1


x = x.reshape(x.shape[0],x.shape[1],1)
print(x.shape) # (90, 4, 1)

from sklearn.model_selection import train_test_split

# train, test 분리할 것.(8:2)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = False, train_size = 0.8)

print(x_train)
print(x_test)
print("=" * 40)
print(x_train.shape)
print(x_test.shape)

# 2. 모델

input1 = Input(shape = (4,1))
# dense1 = LSTM(10, activation= 'relu',return_sequences = True)(input1)
dense2 = Conv1D(10, 2, padding = 'same',activation= 'relu')(input1)
dense2 = Conv1D(10, 2, padding = 'same',activation= 'relu')(dense2)
dense2 = MaxPooling1D()(dense2)
dense2 = Flatten()(dense2)
dense3 = Dense(5,activation='relu')(dense2)

output1 = Dense(10)(dense3)
output2 = Dense(5)(output1)
output3 = Dense(1)(output2)

model = Model(inputs = input1, outputs = output3)

model.summary()
# Conv1D 와 LSMT은 비슷하다. 시계열이나 연속된 데이터가 있을때 사용하면 좋음
# 전자는 특징이 있어보이면 사용하기! 

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

pred = pred.reshape(6,4,1)
y_predict = model.predict(pred)
print(y_predict)



