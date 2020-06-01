# 과제1
# Sequential형으로 완성.

# 하단에 주석으로 acc와 loss결과 명시.
import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


# 1. 데이터, 전처리

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train[0])
print('y_train:', y_train[0]) # 6

print(x_test.shape)  # 10000, 28, 28
print(x_train.shape) # 60000, 28, 28 
print(y_test.shape)  # 10000, 
print(y_train.shape) # 60000, 
print(x_train[0].shape) # 28, 28

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10)
print(y_test.shape)
print(y_train)
print(y_test)

x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.

# 2.모델

model = Sequential()

model.add(Conv2D(50, (3,3), padding = 'same' ,input_shape = (28,28,1)))
model.add(Conv2D(150, (3,3), padding = 'same' ,input_shape = (28,28,1)))
model.add(Dropout(0.4))

model.add(Conv2D(100, (3,3), padding = 'same' ,input_shape = (28,28,1)))
model.add(MaxPooling2D(3,3))

model.add(Conv2D(80, (3,3), padding = 'same' ,input_shape = (28,28,1)))
model.add(MaxPooling2D(3,3))

model.add(Conv2D(50, (3,3), padding = 'same' ,input_shape = (28,28,1)))
model.add(Flatten())

model.add(Dense(100))
model.add(Dense(10, activation= 'softmax'))

model.save('./model/fashion_model.h5')

# 3. 훈련, 컴파일

es = EarlyStopping(monitor= 'loss', patience = 5, mode = 'auto')
modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
mcp =  ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
model.fit(x_train, y_train, epochs = 15, batch_size = 32, validation_split = 0.2, callbacks = [es, mcp])

model.save_weights('./model/fashion_weight.h5')
# 4. 결과

loss, acc = model.evaluate(x_test,y_test)
print('loss:',loss)
print('acc:',acc)

'''
1. loss: 0.3407194226115942, acc: 0.8999999761581421
2. loss: 0.34290156012773515, acc: 0.8956999778747559
3. loss: 0.37627230475842954, acc: 0.8934999704360962

'''