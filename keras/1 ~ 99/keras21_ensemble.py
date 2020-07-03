import numpy as np

# 1. 데이터

x = np.array([range(1,101), range(311,411), range(100)]).T
y = np.array([range(711,811), range(711, 811), range(100)]).T

x2 = np.array([range(101,201), range(411,511), range(100, 200)]).T
y2 = np.array([range(501,601), range(711,811), range(100)]).T
 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = False, train_size = 0.8)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, shuffle = False, train_size = 0.8)

# 2. 모델구성

from keras.models import Sequential, Model
# 함수형 모델을 불러올때는 대문자 M을 사용한 Model
from keras.layers import Dense, Input

input1 = Input(shape = (3, ))
# input은 함수이기에 함수명을 설정한다.  함수에서는 인풋 레이어를 꼭 설정해야한다.
dense1 = Dense(20, activation = 'relu', name = '123')(input1)
# 함수형에서는 맨 뒤에 인풋받은 부분을 명시해야함.
dense2 = Dense(10, activation = 'relu')(dense1)
dense3 = Dense(5, activation = 'relu')(dense2)
dense4 = Dense(4, activation = 'relu')(dense3)

input2 = Input(shape = (3, ))
dense1_1 = Dense(30, activation = 'relu')(input2)
dense2_1 = Dense(20, activation = 'relu')(dense1_1)
dense3_1 = Dense(10, activation = 'relu')(dense2_1)
dense4_1 = Dense(3, activation = 'relu')(dense3_1)

from keras.layers.merge import concatenate
# 레이어에서 멀지에서 콘케트네이트를 불러옴 콘케트네이트는 단순 병합.
merge1 = concatenate([dense4, dense4_1])
# merge1은 두개를 하나로 만들어줌.
middle1 = Dense(30)(merge1)
# 이렇게 하면 아웃풋에 미들이라는 레이어가 또 하나 생기는 것.
middle1 = Dense(5)(merge1)
middle1 = Dense(7)(merge1)
 
 # 이제 아웃풋 모델 구성

output1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(30)(middle1)
output2_2 = Dense(7)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(inputs = [input1, input2], outputs = [output1_3, output2_3])

model.summary()
# 3. 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x_train, x2_train], [y_train, y2_train], epochs = 10, batch_size = 1, validation_split = 0.25, verbose = 1 )
# model.fit(x_train, y_train, epochs = 10, batch_size = 1, validation_split = 0.25, verbose = 1 )
# model.fit(x2_train, y2_train, epochs = 10, batch_size = 1, validation_split = 0.25, verbose = 1 )

# 4. 평가와 예측

loss, loss1, loss2, mse, mse1 = model.evaluate([x_test, x2_test], [y_test, y2_test], batch_size = 1)


print("loss:", loss)
print("loss1:", loss1)
print("loss2:", loss2)
print("mse:", mse)
print("mse1:", mse1)

y_predict, y2_predict = model.predict([x_test, x2_test])
# 여기 테스트는 이미 잘려았어서 20/3
print(y_predict)
print("==============================")
print(y2_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y_test, y_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1:", RMSE1)
print("RMSE2:", RMSE2)
print("RMSE:", (RMSE1 + RMSE2) / 2)


# R2구하기
from sklearn.metrics import r2_score

r2_1 = r2_score(y_test, y_predict)
r2_2 = r2_score(y2_test, y2_predict)

print("R2_1: ", r2_1)
print("R2_2: ", r2_2)
print("R2_2: ", (r2_1+r2_2) / 2)


