
import numpy as np
from sklearn.datasets import load_iris
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()

x = iris.data
y = iris.target

# print(boston)
print(x.shape) # (150, 4)
print(y.shape) # (150, )

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
# print(x)

from keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y.shape) # (150, 3)
print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 256 ,train_size = 0.8)

# print(x_train.shape) # (120, 4)
# print(x_test.shape)  # (30, 4)
# print(y_train.shape) # (120,)
# print(y_test.shape)  # (30,)

# 2. 모델

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape = (4, )))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(5))
model.add(Dense(3, activation = 'softmax'))

model.summary()

# 3. 훈련, 컴파일

es = EarlyStopping(monitor='val_loss', patience= 20, mode = 'auto')

modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
                        mode = 'auto', save_best_only = True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs =100, batch_size= 64,
                validation_split = 0.2, verbose = 2,
                callbacks = [es, mcp])


#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
print('loss', loss)
print('acc', acc)

# 시각화

import matplotlib.pyplot as plt
plt.figure(figsize = (10, 5))
# 사이즈 조절, 크기 단위는

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], c= 'red', marker = '.', label = 'loss')
plt.plot(hist.history['val_loss'], c= 'blue', marker = '.', label = 'val_loss')
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], c= 'red', marker = '.', label = 'acc')
plt.plot(hist.history['val_acc'], c= 'blue', marker = '.', label = 'val_acc')
plt.title('accuarcy')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()