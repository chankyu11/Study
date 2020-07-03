from numpy import array 
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터구성

x = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
           [5,6,7], [6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = array([50,60,70])


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

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_length = 3, input_dim = 1,
               return_sequences = True))
model.add(LSTM(10, return_sequences = False))
input1 = Input(shape = (3,1))
# xlstm1 = LSTM(10, return_sequences = True)(input1)
# return_sequences를 하면 차원을 유지시켜준다.
input1 = Input(shape = (3,1))
xlstm1 = (LSTM(10, input_length = 3, input_dim = 1, return_sequences= True))(input1)
xlstm2 = LSTM(10)(xlstm1)
dense1 = Dense(5)(xlstm2)
dense2 = Dense(1)(dense1)

# input2 = Input(shape = (3,1))
# dense1_1 = LSTM(80, activation = 'relu', input_shape = (3,1))(input2)
# # model.add(LSTM(15, input_length = 3, input_dim = 1))
# dense2_1 = Dense(50)(dense1_1)
# dense3_1 = Dense(30)(dense2_1)
# dense4_1 = Dense(20)(dense3_1)
# dense5_1 = Dense(10)(dense4_1)
# dense6_1 = Dense(3)(dense5_1)

# from keras.layers.merge import concatenate
# merge1 = concatenate([dense2, dense6_1])
# middle = Dense(70)(merge1)
# middle1 = Dense(60)(middle)
# middle2 = Dense(50)(middle1)
# middle3 = Dense(30)(middle2)
# middle4 = Dense(10)(middle3)

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

x_predict = array([50, 60, 70])          
x_predict = x_predict.reshape(1, 3, 1)
print(x_predict)

y_predict = model.predict(x_predict)
print("y_predict:", y_predict)
