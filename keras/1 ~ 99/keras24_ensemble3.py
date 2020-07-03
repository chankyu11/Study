import numpy as np

# 1. 데이터

x1 = np.array([range(1, 101), range(301, 401)]).T

y1 = np.array([range(711, 811), range(611, 711)]).T
y2 = np.array([range(101, 201), range(411, 511)]).T

from sklearn.model_selection import train_test_split

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, shuffle = False, train_size = 0.8)

print(x_train.shape) # 80/ 2
print(y1_test.shape) # 20/ 2

# 2. 모델구성

from keras.models import Sequential, Model
# 함수형 모델을 불러올때는 대문자 M을 사용한 Model

from keras.layers import Dense, Input

input1 = Input(shape = (2, ))
dense1 = Dense(5, activation = 'relu', name = 'a')(input1)
dense2 = Dense(5, activation = 'relu', name = 'b')(dense1)
dense3 = Dense(5, activation = 'relu', name = 'c')(dense2)
dense4 = Dense(2, activation = 'relu', name = 'd')(dense3)

# input2 = Input(shape = (2, ))
# dense1_1 = Dense(5, activation = 'relu', name = 'e')(input2)
# dense2_1 = Dense(5, activation = 'relu', name = 'f')(dense1_1)
# dense3_1 = Dense(3, activation = 'relu', name = 'g')(dense2_1)
# dense4_1 = Dense(2, activation = 'relu', name = 'h')(dense3_1)

# from keras.layers.merge import concatenate

# merge1 = concatenate([dense4, dense4_1])
# middle1 = Dense(7, name = 'i')(merge1)
# middle1 = Dense(5, name = 'j')(merge1)
# middle1 = Dense(1, name = 'k')(merge1)
 
#  이제 아웃풋 모델 구성

output1 = Dense(10, name = 'l')(dense4)
output1_2 = Dense(3, name = 'm')(output1)
output1_3 = Dense(2, name = 'n')(output1_2)

output2 = Dense(30)(dense4)
output2_2 = Dense(7)(output2)
output2_3 = Dense(2)(output2_2)

# # output3 = Dense(30)(middle1)
# # output3_2 = Dense(7)(output3)
# # output3_3 = Dense(3)(output3_2)

model = Model(inputs = input1, outputs = [output1_3, output2_3])

model.summary()

# 3. 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, [y1_train, y2_train], epochs = 10, batch_size = 1, validation_split = 0.25, verbose = 1 )

# 4. 평가와 예측

loss, loss1, loss2, mse, mse1 = model.evaluate(x_test, [y1_test, y2_test], batch_size = 1)

print("loss:", loss)
print("loss1:", loss1)
print("loss2:", loss2)
# print("loss3:", loss3)
print("mse:", mse)
print("mse1:", mse1)
# print("mse2:", mse2)

y1_predict, y2_predict = model.predict(x_test)
print(y1_predict)
print("==============================")
print(y2_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 :", RMSE1)
print("RMSE2 :", RMSE2)
print("RMSE :", (RMSE1 + RMSE2) / 2)

# R2구하기
from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)

print("R2: ", r2_1)
print("R2: ", r2_2)
print("R2: ", (r2_1 + r2_2) /2)
