#1. 데이터

import numpy as np

x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

#2. 모델구성

from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

input1 = Input(shape=(1,))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape = (1,))
x2 = Dense(50)(input2)
x2 = Dense(50)(x2)
x2 = Dense(50)(x2)

merge = concatenate([x1,x2])

x3 = Dense(25)(merge)
output1 = Dense(1)(x3)

x4 = Dense(25)(merge)
output2 = Dense(1, activation = 'sigmoid')(x4)

model = Model(input = [input1,input2], outputs = [output1, output2])

model.summary()

#3. 컴파일, 훈련

model.compile(loss = ['mse', 'binary_crossentropy'], 
              metrics= ['mse', 'acc'],
            #   loss_weights=[0.1, 0.9],  
              optimizer = 'adam')
# loss에 대한 가중치를 0.1은 앞에, 0.9를 후자에 주겠다.
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs = 100, batch_size=1)

#4. 평가

loss = model.evaluate([x1_train, x2_train], [y1_train, y2_train])

print("loss: ", loss)

x1_pred = np.array([11,12,13,14])
x2_pred = np.array([11,12,13,14])

y_pred = model.predict([x1_pred, x2_pred])
print(y_pred)