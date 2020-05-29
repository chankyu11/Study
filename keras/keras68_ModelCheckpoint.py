import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Conv2D , MaxPooling2D, Dense, Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape) # (60000, 10)
# print(y_test.shape)
# print(y_train)
# print(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 2. 모델

model = Sequential()
model.add(Conv2D(100, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# model.add(MaxPooling2D(3,3))
model.add(Conv2D(50, (2,2), padding = 'same' ,input_shape = (28,28,1)))
model.add(Conv2D(25, (2,2), padding = 'same' ,input_shape = (28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(13, (2,2), padding = 'same' ,input_shape = (28,28,1)))
model.add(Flatten())
model.add(Dense(23))
model.add(Dense(10, activation= 'softmax'))

# 2. 모델

model = Sequential()
model.add(Conv2D(100, (2,2), padding = 'same' ,input_shape = (28,28,1)))
# model.add(MaxPooling2D(3,3))
model.add(Conv2D(50, (2,2), padding = 'same' ,input_shape = (28,28,1)))
model.add(Conv2D(25, (2,2), padding = 'same' ,input_shape = (28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(13, (2,2), padding = 'same' ,input_shape = (28,28,1)))
model.add(Flatten())
model.add(Dense(23))
model.add(Dense(10, activation= 'softmax'))
model.summary()

# 3. 훈련, 컴파일

modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
# {epoch:02d} d는 정수, 두자리 정수, {val_loss:.4f} 4자리 실수. 확장자는 hdf5
mcp =  ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto')
# 모델의 경로는 model. val_loss로 모니터, 세이브 베스트 원 = 가장 좋은 걸 저장.
es = EarlyStopping(monitor = 'loss', patience = 5)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
# model.fit(x_train, y_train, validation_split=0.2, epochs = 15, batch_size = 32, callbacks= ['es'])
hist = model.fit(x_train,y_train, validation_split = 0.2 ,epochs = 10 ,verbose=1, 
                callbacks = [es, mcp])



# 4. 평가 예측

loss, acc = model.evaluate(x_test,y_test)
print('loss:',loss)
print('acc:',acc)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc =  hist.history['val_accuracy']

print('acc: ', acc)
print('val_acc: ', val_acc)

plt.figure(figsize=(10 ,6))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker=',', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker=',', c='blue', label='val_loss')
#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right') 
plt.show()

plt.subplot(2, 1, 2)
#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.grid() #  그림에  画格子
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy','val_accuracy'])

plt.show()




