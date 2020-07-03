from numpy import array 
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

'''
minmax 스칼라 = 범위는 0 ~ 1
minmax_scale 
범위를 줄임.
한쪽으로 편향된 데이터를 정규분포로 만들어주는 스탠다드 스칼라.
'''
# 1. 데이터구성

x = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
           [5,6,7], [6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],
           [100,200,300]])

y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])

x_predict = array([55,65,75])
print(x.shape)
print(y.shape)
print(x_predict.shape)

# from sklearn.preprocessing import MinMaxScaler
# # sklearn에서 preprocessing 불러오고 민맥스 스칼라 소환!
# scaler = MinMaxScaler()
# scaler.fit(x)
# # 스칼라를 실행하겠다. 
# x = scaler.transform(x)
# # 꼭 fit 진행 후 transform
# print(x)
# x = x.reshape(x.shape[0], x.shape[1], 1)
# print(x.shape)

x_predict = x_predict.reshape(1, 3)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x)
x = scaler.transform(x)
print(x)

print("===================================")

scaler.fit(x_predict)
x_predict = scaler.transform(x_predict)
print(x_predict)


# # print("x: ", x.shape) # (13, 3)
# # print("y: ", y.shape) #(3, )

# # # x = x.reshape(4,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1) # 위하고 똑같음
x_predict = x_predict.reshape(1, 3, 1)

# #                 행          열    몇개씩 자르는지
# # x의 shape = (batch_size, timesteps, feature)
# # input_shape = (timesteps, feature)
# # input_length = (timesteps, input_dim = feature) 


# 2. 모델구성

# model = Sequential()
# model.add(LSTM(10, activation = 'relu', input_length = 3, input_dim = 1,
            #    return_sequences = True))
# model.add(LSTM(10, return_sequences = False))
# xlstm1 = LSTM(10, return_sequences = True)(input1)
# return_sequences를 하면 차원을 유지시켜준다.
input1 = Input(shape = (3,1))
xlstm1 = (LSTM(100, input_length = 3, input_dim = 1, return_sequences= True))(input1)
xlstm2 = LSTM(100)(xlstm1)
dense1 = Dense(50)(xlstm2)
dense2 = Dense(1)(dense1)

output1 = Dense(20)(dense2)
output2 = Dense(10)(output1)
output3 = Dense(5)(output2)
output4 = Dense(1)(output3)

model = Model(inputs = input1, outputs = output4)

model.summary()

# 3. 실행

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x, y, epochs = 950, batch_size = 32, callbacks = [es] )

# 4. 예측

# x1_predict = array([55,65,75])
# x2_predict = array([65,75,85])

# x_predict = array([55, 65, 75])          
# x_predict = x_predict.reshape(1, 3)
# scaler.fit(x_predict)
# x_predict = scaler.transform(x_predict)
print(x_predict)

y_predict = model.predict(x_predict)
print("y_predict:", y_predict)
