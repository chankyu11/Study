import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM
from keras.callbacks import EarlyStopping

# 2. 모델

model = Sequential()
model.add(LSTM(10, input_shape = 4, input_dim = 1))
model.add(Dense(15))
model.add(Dense(25))
model.add(Dense(35))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(10))

model.summary

model.save(".//model//save_keras44.h5")
# model.save("./model/save_keras44.h5")
# model.save(".\model\save_keras44.h5")
# 3가지 모두 가능! .//, ./, .\ 저장공간 이름 , / 파일명

print("저장 끝")