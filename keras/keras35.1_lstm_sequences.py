from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input


# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = array([50,60,70]) # (3,)

print("x.shape:", x.shape) # (13, 3)
print("y.shape:", y.shape) # (13,)

# x = x.reshape(4, 3, 1)                      
x = x.reshape(x.shape[0], x.shape[1], 1) # (13, 3, 1)
print(x.shape)


# 2. 모델구성
input1 = Input(shape=(3,1))
dense1_1 = LSTM(20, activation='relu', name='dense1_1',
                return_sequences=True)(input1)
dense1_2 = LSTM(10, activation='relu', name='dense1_2')(dense1_1)
dense1_3 = Dense(5, activation='relu', name='dense1_3')(dense1_2)
# dense1_3 = Dense(15, activation='relu', name='dense1_3')(dense1_2)
# dense1_4 = Dense(11, activation='relu', name='dense1_4')(dense1_3)
# dense1_5 = Dense(6, activation='relu', name='dense1_5')(dense1_4)
output1 = Dense(1, name='output1')(dense1_3)

model = Model(inputs=input1, outputs=output1)

model.summary()

# lstm 의 형태 x=(행,열,몇) 즉, (batch,time,feature) : x는 3차원
# LSTM(n,) = dense 레이어 자체는 2차원 (행,열)만 입력 받기 때문에
# 윗 레이어의 아웃풋이 2차원, 근데 lstm이라 3차원을 받아야 해
# 즉, "3차원으로 바꿔라" 라는 의미.
# return_sequences 을 사용하면 차원을 유지시켜준다.
# Dense 모델은 2차원을 받아들이기 때문에 위에 return_sequences를 사용하면 역시나 오류가 뜬다.

# lstm 2번째 레이어 파라미터 개수 구하는 공식 2가지
# 1 = 2
# 1. [(num_units + input_dim + 1) * num_units] * 4 = num_params
# [(10 + 10 + 1) * 10] * 4 = 840
# 2. [(size_of_input + 1) * size_of_output + size_of_output^2] * 4 = num_params
# [(10 + 1) * 10 + 100] * 4 = 840
# input_dim이 10인 이유
''' 
내 정리
    h(t-1), h(t), h(t2)... h(t9) = 10
(lstm1) o o o o o o o o o o

    h(t-1), h(t), h(t2)... h(t9) = 10
-> input_dim 으로 연산 = 10
(lstm2) o o o o o o o o o o
샘 정리
아웃풋 노드의 개수가 feature로 들어간다.
'''


# 3. 훈련
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.fit(x, y, epochs = 800, batch_size = 16, callbacks = [early_stopping])

x_predict = x_predict.reshape(1,3,1)
print(x_predict)

# 4. 예측

y_predict = model.predict(x_predict)
print(y_predict)
