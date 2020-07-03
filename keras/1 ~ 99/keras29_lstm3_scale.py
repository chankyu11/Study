from numpy import array 
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터구성

x = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
           [5,6,7], [6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# x_predict = array([50,60,70])

# print("x: ", x.shape) # shape = (4, 3)
# print("y: ", y.shape) # shape = (4, ) = 스칼라가 4개 짜리라는 뜻

# 제일 겉에 괄호는 무시하고 생각하면 편하고, 안에서부터 数

# x = x.reshape(4,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1) # 위하고 똑같음

'''
                행          열    몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = (timesteps, input_dim = feature) 

'''
# print(x.shape)
# print(x)


# 2. 모델구성

model = Sequential()
model.add(LSTM(50, activation = 'relu', input_shape = (3,1)))
# =LSTM(10)은 시작 노드가 10개라는 뜻
# 4,3,1 인데 행은 무시 그렇기에 3,1 3개씩 잘라서, 1개씩 넣겠다는 의미
# model.add(LSTM(15, input_length = 3, input_dim = 1))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

# 3. 실행

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x, y, epochs = 500, batch_size = 32)

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

