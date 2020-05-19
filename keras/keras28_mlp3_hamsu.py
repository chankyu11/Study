
import numpy as np

# 1. 데이터

x = np.array([range(1,101), range(311,411), range(100)]).T
y = np.array([range(711,811)]).T

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, train_size = 0.8)

# 2. 모델구성

from keras.models import Model
from keras.layers import Dense, Input

input1 = Input(shape = (3, ))
dense1 = Dense(10, activation = 'relu')(input1)
dense3 = Dense(10, activation = 'relu')(dense1)
dense6 = Dense(3, activation = 'relu')(dense3)

output1 = Dense(3)(dense6)
output2 = Dense(10)(output1)
output3 = Dense(5)(output2)
output1_3 = Dense(1)(output3)


model = Model(inputs = input1, outputs = output1_3)

# 3. 훈련

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 1000, batch_size = 1, validation_split = 0.25, callbacks = [early_stopping])

# 4. 평가와 예측

loss, mse = model.evaluate(x_test, y_test, batch_size = 8)

print("loss:", loss)
print("mse:", mse)

# y 예측값
y_predict = model.predict(x_test)
print("y_predict: ", y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

# R2구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)

