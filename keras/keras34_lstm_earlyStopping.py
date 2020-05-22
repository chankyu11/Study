# 앙상블로 바꾸시오.

from numpy import array 
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터구성

x1 = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
           [5,6,7], [6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])

x2 = array([[10,20,30],[20,30,40], [30,40,50],[40,50,60],
           [50,60,70], [60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[100,110,120],
           [2,3,4],[3,4,5],[4,5,6]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65,75])
x2_predict = array([65,75,85])


print("x1: ", x1.shape) # (13, 3)
print("x2: ", x2.shape) # (13, 3)

print("y: ", y.shape) #(3, )

# x = x.reshape(4,3,1)
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) # 위하고 똑같음
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) # 위하고 똑같음

'''
                행          열    몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = (timesteps, input_dim = feature) 

'''

# 2. 모델구성

input1 = Input(shape = (3,1))
dense1 = LSTM(80, activation = 'relu', input_shape = (3,1))(input1)
# model.add(LSTM(15, input_length = 3, input_dim = 1))
dense2 = Dense(50)(dense1)
dense3 = Dense(30)(dense2)
dense4 = Dense(20)(dense3)
dense5 = Dense(10)(dense4)
dense6 = Dense(3)(dense5)

input2 = Input(shape = (3,1))
dense1_1 = LSTM(80, activation = 'relu', input_shape = (3,1))(input2)
# model.add(LSTM(15, input_length = 3, input_dim = 1))
dense2_1 = Dense(50)(dense1_1)
dense3_1 = Dense(30)(dense2_1)
dense4_1 = Dense(20)(dense3_1)
dense5_1 = Dense(10)(dense4_1)
dense6_1 = Dense(3)(dense5_1)

from keras.layers.merge import concatenate
merge1 = concatenate([dense6, dense6_1])
middle = Dense(70)(merge1)
middle1 = Dense(60)(middle)
middle2 = Dense(50)(middle1)
middle3 = Dense(30)(middle2)
middle4 = Dense(10)(middle3)

output1 = Dense(20)(middle4)
output2 = Dense(10)(output1)
output3 = Dense(5)(output2)
output4 = Dense(1)(output3)

model = Model(inputs = [input1,input2], outputs = output4)

model.summary()

# 3. 실행
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')

model.compile(optimizer = 'adam', loss = 'mse')
model.fit([x1,x2], y, epochs = 950, batch_size = 32, callbacks = [es] )

# 4. 예측

# x1_predict = array([55,65,75])
# x2_predict = array([65,75,85])

x1_predict = array([55, 65, 75])          
x1_predict = x1_predict.reshape(1, 3, 1)
print(x1_predict)

x2_predict = array([65, 75, 85])          
x2_predict = x2_predict.reshape(1, 3, 1)
print(x2_predict)

y_predict = model.predict([x1_predict, x2_predict])
print("y_predict:", y_predict)
