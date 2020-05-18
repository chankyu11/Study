import numpy as np

# 1. 데이터

x1 = np.array([range(1,101), range(311,411)]).T
x2 = np.array([range(711,811), range(711, 811)]).T

y1 = np.array([range(101,201), range(411,511)]).T
y2 = np.array([range(501,601), range(711,811)]).T
y3 = np.array([range(411,511), range(611,711)]).T

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x1, y1, shuffle = False, train_size = 0.8)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, shuffle = False, train_size = 0.8)
y3_train, y3_test = train_test_split(y3, shuffle = False, train_size = 0.8)

# 2. 모델구성

from keras.models import Sequential, Model
# 함수형 모델을 불러올때는 대문자 M을 사용한 Model
from keras.layers import Dense, Input

input1 = Input(shape = (2, ))
# input은 함수이기에 함수명을 설정한다.  함수에서는 인풋 레이어를 꼭 설정해야한다.
dense1 = Dense(20, activation = 'relu', name = 'a')(input1)
# 함수형에서는 맨 뒤에 인풋받은 부분을 명시해야함.
dense2 = Dense(10, activation = 'relu', name = 'b')(dense1)
dense3 = Dense(5, activation = 'relu', name = 'c')(dense2)
dense4 = Dense(4, activation = 'relu', name = 'd')(dense3)

input2 = Input(shape = (2, ))
dense1_1 = Dense(30, activation = 'relu', name = 'e')(input2)
dense2_1 = Dense(20, activation = 'relu', name = 'f')(dense1_1)
dense3_1 = Dense(10, activation = 'relu', name = 'g')(dense2_1)
dense4_1 = Dense(3, activation = 'relu', name = 'h')(dense3_1)

from keras.layers.merge import concatenate
merge1 = concatenate([dense4, dense4_1])
middle1 = Dense(30)(merge1)
middle1 = Dense(30)(merge1)
middle1 = Dense(5)(merge1)
middle1 = Dense(7)(merge1)
 
 # 이제 아웃풋 모델 구성

output1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1)
output1_3 = Dense(2)(output1_2)

output2 = Dense(30)(middle1)
output2_2 = Dense(7)(output2)
output2_3 = Dense(2)(output2_2)

output3 = Dense(30)(middle1)
output3_2 = Dense(7)(output3)
output3_3 = Dense(2)(output3_2)

model = Model(inputs = [input1, input2], outputs = [output1_3, output2_3, output3_3])

model.summary()

# 3. 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x_train, x2_train], [y_train, y2_train, y3_train], epochs = 200, batch_size = 1, validation_split = 0.25, verbose = 1 )

# 4. 평가와 예측

loss, loss1, loss2, loss3, mse, mse1, mse2 = model.evaluate([x_test, x2_test], [y_test, y2_test, y3_test], batch_size = 1)


print("loss:", loss)
print("loss1:", loss1)
print("loss2:", loss2)
print("loss3:", loss3)
print("mse:", mse)
print("mse1:", mse1)
print("mse2:", mse2)

y_predict, y2_predict, y3_predict = model.predict([x_test, x2_test])
# 여기 테스트는 이미 잘려았어서 20/3
print(y_predict)
print("==============================")
print(y2_predict)
print("==============================")
print(y3_predict)
print("==============================")


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y_test, y_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)

print("RMSE1:", RMSE1)
print("RMSE2:", RMSE2)
print("RMSE3:", RMSE3)
print("RMSE:", (RMSE1 + RMSE2 + RMSE3) / 3)


# R2구하기
from sklearn.metrics import r2_score

r2_1 = r2_score(y_test, y_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2_3 = r2_score(y3_test, y3_predict)

print("R2_1: ", r2_1)
print("R2_2: ", r2_2)
print("R2_3: ", r2_3)
print("R2_2: ", (r2_1 + r2_2 + r2_3) / 3)


