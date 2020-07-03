import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Conv2D , MaxPooling2D, Dense, Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
# print("=" * 20)
# print('y_train: ', y_train[0])
# print(y_train.shape)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10)
print(y_test.shape)
print(y_train)
print(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
# x_train의 shape를 4차원을 만들고, 그걸 float 타입으로 변환하고 거기에 除255
# minmax 는 0 ~ 1로 나오기에 값을 실수로 만들어야함.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# # 2. 모델

# model = Sequential()
# model.add(Conv2D(100, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# # model.add(MaxPooling2D(3,3))
# model.add(Conv2D(50, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# model.add(Conv2D(25, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(13, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# model.add(Flatten())
# model.add(Dense(23))
# model.add(Dense(10, activation= 'softmax'))
# model.summary()

# # model.save('./model/model_test01.h5')

# # 3. 훈련, 컴파일

# es = EarlyStopping(monitor='loss', patience=5)
# modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
# mcp =  ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
#                      save_best_only = True, save_weights_only= False,
#                      mode = 'auto', verbose = 1)
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
# # model.fit(x_train, y_train, validation_split=0.2, epochs = 15, batch_size = 32, callbacks= ['es'])
# hist = model.fit(x_train,y_train, validation_split = 0.1 ,
#         epochs = 2, verbose=1, callbacks = [es, mcp])

# model.save('./model/model_test01.h5')

from keras.models import load_model
model = load_model('./model/02 - 0.0839.hdf5')
# 4. 평가 예측

loss_acc = model.evaluate(x_test,y_test)

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']

# print('acc:', acc)
# print('val_acc:', val_loss)
print('loss_acc:', loss_acc)

# plt.figure(figsize=(10,6))

# plt.subplot(2,1,1) # 두장의 그림을 그리겠다. (2행, 1열의 그림을 그릴꺼, 첫 그림을 그릴꺼)

# plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
# plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
# # plt.plot(hist.history['acc'])
# # plt.plot(hist.add_history['val_acc'])
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right') # 명시된 자리.

# plt.subplot(2,1,2) 
# # 두장의 그림을 그리겠다. (2행, 1열의 그림을 그릴꺼, 두번째 그림을 그릴꺼)

# # plt.plot(hist.history['loss'])
# # plt.plot(hist.history['val_loss'])
# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.grid() #  그림에  画格子
# plt.title('accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['accuracy','val_accuracy'])

# plt.show()