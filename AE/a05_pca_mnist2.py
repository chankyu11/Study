import numpy as np
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#1. 데이터

(x_train, y_train), (x_test, y_test) =  mnist.load_data()
# print(x_train.shape)        # (60000, 28, 28)
# print(x_test.shape)         # (10000, 28, 28)
# print(y_train.shape)        # (60000,)
# print(y_test.shape)         # (10000,)

X = np.append(x_train, x_test, axis = 0)
X = X.reshape(-1,28*28)

print(X.shape)

pca = PCA(n_components=154)
pca.fit(X)
x = pca.transform(X)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(cumsum)

# n_components = np.argmax(cumsum >= 0.99) + 1
# print(n_components)
x_train = x[:60000]
x_test = x[60000:]

# print(x_train.shape)
# print(x_test.shape)

# 1-1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)        # (60000, 10)
print(y_test.shape)         # (10000, 10)

x_train = x_train.reshape(-1, 154).astype('float32')/255
x_test = x_test.reshape(-1, 154).astype('float32')/255

print(x_train.shape)        # (60000, 154)
print(x_test.shape)         # (10000, 154)

#2. 모델구성

model = Sequential()
model.add(Dense(77, input_dim = (154)))
model.add(Dense(35))
model.add(Dense(10, activation = 'softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

#4. 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)