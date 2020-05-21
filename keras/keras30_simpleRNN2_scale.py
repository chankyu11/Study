from numpy import array 
# import numpy as np
# x = np.array
# from numpy import array와 위 두 줄의 의미는 같다.
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# 1. 데이터구성

x = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
           [5,6,7], [6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# 1. [[1,2,3],[1,2,3]] = 2,3 2차원 텐서
# 2. [[1,2], [4,3]],[[4,5],[5,6]] = 2,2,2 3차원 텐서 
# 3. [[[1],[2],[3]],[[4],[5],[6]]] = 2, 3, 1
# 4. [[[1,2,3,4]]] = 2, 3, 1
# 5. [[[[1],[2]]],[[[3],[4]]]] = 2,1,4
# 제일 겉에 괄호는 무시하고 생각하면 편하고, 안에서부터 数

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
model.add(SimpleRNN(10, activation = 'relu', input_shape = (3,1)))
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

x_predict = array([50, 60, 70])               # (3, )
x_predict = x_predict.reshape(1, 3, 1)       # x값 (4, 3, 1)와 동일한 shape로 만들어 주기 위함
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)