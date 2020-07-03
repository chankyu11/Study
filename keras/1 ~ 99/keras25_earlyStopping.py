import numpy as np

# 1. 데이터

x1 = np.array([range(1,101), range(311,411), range(411,511)]).T
x2 = np.array([range(711,811), range(711, 811), range(511, 611)]).T

y1 = np.array([range(101,201), range(411,511), range(100)]).T

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test, x2_train, x2_test = train_test_split(x1, y1, x2, train_size = 0.8)

# 2. 모델구성

from keras.models import Sequential, Model
# 함수형 모델을 불러올때는 대문자 M을 사용한 Model
from keras.layers import Dense, Input

input1 = Input(shape = (3, ))
# input은 함수이기에 함수명을 설정한다.  함수에서는 인풋 레이어를 꼭 설정해야한다.
dense1 = Dense(5, activation = 'relu')(input1)
# 함수형에서는 맨 뒤에 인풋받은 부분을 명시해야함.
dense2 = Dense(10, activation = 'relu')(dense1)
dense3 = Dense(20, activation = 'relu')(dense2)
dense4 = Dense(10, activation = 'relu')(dense3)
dense5 = Dense(5, activation = 'relu')(dense4)
dense6 = Dense(3, activation = 'relu')(dense5)
# print(dense4)

input2 = Input(shape = (3, ))
dense1_1 = Dense(5, activation = 'relu')(input2)
dense2_1 = Dense(10, activation = 'relu')(dense1_1)
dense3_1 = Dense(20, activation = 'relu')(dense2_1)
dense4_1 = Dense(10, activation = 'relu')(dense3_1)
dense5_1 = Dense(5, activation = 'relu')(dense4_1)
dense6_1 = Dense(3, activation = 'relu')(dense5_1)

# print(dense4_1)

from keras.layers.merge import concatenate
merge1 = concatenate([dense6, dense6_1])
middle1 = Dense(5)(merge1)
middle1 = Dense(10)(merge1)
middle1 = Dense(20)(merge1)
middle1 = Dense(10)(merge1)
middle1 = Dense(5)(merge1)
middle1 = Dense(3)(merge1)
# print(middle1)
 
#  이제 아웃풋 모델 구성

output1 = Dense(10)(middle1)
output1_2 = Dense(5)(output1)
output1_3 = Dense(3)(output1_2)
# print(output1_3)

# output2 = Dense(30)(middle1)
# output2_2 = Dense(7)(output2)
# output2_3 = Dense(3)(output2_2)

# output3 = Dense(30)(middle1)
# output3_2 = Dense(7)(output3)
# output3_3 = Dense(3)(output3_2)

model = Model(inputs = [input1, input2], outputs = output1_3)

# model.summary()

# 3. 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor= 'loss', patience = 10, mode = 'min')
# 얼리스타핑에 모니터하겠다 그걸 loss로 하겠다. 페이션스 횟수는 10번이다. 모드는 min 최소값이다.

model.fit([x_train, x2_train], y_train, epochs = 10000, batch_size = 1, validation_split = 0.25, verbose = 1, callbacks = [early_stopping] )

# 4. 평가와 예측

loss = model.evaluate([x_test, x2_test], y_test, batch_size = 1)

print("loss:", loss)
# # print("loss1:", loss1)
# # print("loss2:", loss2)
# # print("loss3:", loss3)
# print("mse:", mse)
# # print("mse1:", mse1)
# # print("mse2:", mse2)

y_predict = model.predict([x_test, x2_test])
print("y_predict :", y_predict)
print("==============================")

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE = RMSE(y_test, y_predict)

print("RMSE:", RMSE)

# R2구하기
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)

print("R2: ", r2)


