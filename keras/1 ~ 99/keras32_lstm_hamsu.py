# 함수령으로 바꾸시오.

from numpy import array 
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터구성

x = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
           [5,6,7], [6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# x_predict = array([50,60,70])

print("x: ", x.shape) # (13, 3)
print("y: ", y.shape) #(3, )

# x = x.reshape(4,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1) # 위하고 똑같음

'''
                행          열    몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = (timesteps, input_dim = feature) 

'''

# 2. 모델구성
input1 = Input(shape = (3,1))
dense1 = LSTM(50, activation = 'relu', input_shape = (3,1))(input1)
# model.add(LSTM(15, input_length = 3, input_dim = 1))
dense2 = Dense(30)(dense1)
dense3 = Dense(20)(dense2)
dense4 = Dense(10)(dense3)
dense5 = Dense(5)(dense4)
dense6 = Dense(3)(dense5)

output1 = Dense(10)(dense6)
output2 = Dense(5)(output1)
output3 = Dense(1)(output2)

model = Model(inputs = input1, outputs = output3)

model.summary()

# 3. 실행

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x, y, epochs = 800, batch_size = 32)

# 4. 예측

x_predict = array([50, 60, 70])          
x_predict = x_predict.reshape(1, 3, 1)
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)

# x_input = array([5,6,7])
# print(x_input)
# x_input = x_input.reshape(1,3,1)
# print(x_input)

# yhat = model.predict(x_input)
# print(yhat)

