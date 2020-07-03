from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
mcp =  ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto')
es = EarlyStopping(monitor= 'loss', patience = 5, mode = 'auto')
# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train[0])
print('y_train:', y_train[0])

print(x_test.shape)     # (10000, 32, 32, 3)
print(x_train.shape)    # (50000, 32, 32, 3)
print(y_test.shape)     # (10000, 1)
print(y_train.shape)    # (50000, 1)
print(x_train[0].shape) # (32, 32, 3)

# 1.1 데이터 원핫인코딩

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (50000, 10)
print(y_test.shape)  # (10000, 100)
print(y_train)
print(y_test)

# 1.2 정규화

x_test = x_test.reshape(10000, 256, 12).astype('float32') / 255
x_train = x_train.reshape(50000, 256, 12).astype('float32') / 255
print(x_train.shape)
print(x_test.shape)

# 2. 모델

input1 = Input(shape=(256,12))
dense1 = LSTM(200, activation='relu', return_sequences= 'True')(input1)
dense2 = LSTM(100,activation='relu')(dense1)
dense3 = Dense(50,activation='relu')(dense2)

output1 = Dense(200)(dense3)
output2 = Dense(50)(output1)
output3 = Dense(100, activation='softmax')(output2)

model = Model(inputs = input1, outputs = output3)

model.summary()

# 3. 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs = 15,batch_size = 32)

# 4. 평가 

loss_acc = model.evaluate(x_test, y_test)
print("result: ", loss_acc)

# 5. 시각화
plt.figure(figsize = (10, 6))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.title('loss')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = '.', c = 'red', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '.', c = 'blue', label = 'val_acc')
plt.title('accuracy')

plt.grid()
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()
