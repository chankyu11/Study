# 함수령으로 바꾸시오.

from numpy import array 
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터구성

x1 = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
           [5,6,7], [6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65,75])

print('x1.shape : ',x1.shape)# (13, 3)
print('y.shape : ',y.shape)# (13, ) 

# x = x.reshape(4,3,1)
# x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) # 위하고 똑같음
# x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

'''
                행          열    몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = (timesteps, input_dim = feature) 

'''

# 2. 모델구성

input1 = Input(shape = (3, ))
dense1 = Dense(50, activation = 'relu')(input1)
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

'''
LSTM_parameter 계산
num_params = 4 * ( num_units   +   input_dim   +   1 )  *  num_units
                (output node값)  (잘라준 data)   (bias)  (output node값)
           = 4 * (    5      +       1       +   1 )  *     5          = 140     
                    역전파 : 나온 '출력' 값이 다시 '입력'으로 들어감(자귀회귀)
'''

# 3. 실행

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x1, y, epochs = 800, batch_size = 32)

# 4. 예측

x1_predict = array([55, 65, 75])          
x1_predict = x1_predict.reshape(1, 3, )
print(x1_predict)

y_predict = model.predict(x1_predict)
print("y_predict:", y_predict)

