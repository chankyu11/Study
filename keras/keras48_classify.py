import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM
from keras.callbacks import EarlyStopping

# sigmoid
#  분류모델에서 사용되는 손실함수

# 1. 데이터

x= np.array(range(1,11))
y= np.array([1,0,1,0,1,0,1,0,1,0])

# 2. 모델

model = Sequential()
model.add(Dense(10, activation = 'relu', input_dim = 1))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# 3. 훈련, 컴파일

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics= ['acc'])
# 단일분류는  binary_crossentropy,  activation = sigmoid
model.fit(x, y, epochs = 100, batch_size = 1)

# 4. 평가 예측

loss, acc = model.evaluate(x, y, batch_size = 1)
print("loss:", loss)
print("acc:", acc)

x_pred = np.array([1,2,3])
y_pred = model.predict(x_pred)
print('y_pred:', y_pred)


# y_pred가 0.5 이상이면 1을 출력 아니면 0을 출력
for i in y_pred:
    if i >= 0.5:
        print(1)
    else:
        print(0)