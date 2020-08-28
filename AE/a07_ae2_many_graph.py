import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import random

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, input_shape = (784, ), activation = 'relu'))
    model.add(Dense(units = 784, activation = 'sigmoid'))
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()

x_train, y_train = train_set
x_test, y_test = test_set

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]) / 255.

# print(x_train.shape)
# print(x_test.shape)

model01 = autoencoder(hidden_layer_size = 1)
model02 = autoencoder(hidden_layer_size = 2)
model04 = autoencoder(hidden_layer_size = 4)
model08 = autoencoder(hidden_layer_size = 8)
model16 = autoencoder(hidden_layer_size = 16)
model32 = autoencoder(hidden_layer_size = 32)

model01.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model01.fit(x_train, x_train, epochs = 10)

model02.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model02.fit(x_train, x_train, epochs = 10)

model04.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model04.fit(x_train, x_train, epochs = 10)

model08.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model08.fit(x_train, x_train, epochs = 10)

model16.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model16.fit(x_train, x_train, epochs = 10)

model32.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model32.fit(x_train, x_train, epochs = 10)


op01 = model01.predict(x_test)
op02 = model02.predict(x_test)
op04 = model04.predict(x_test)
op08 = model08.predict(x_test)
op16 = model16.predict(x_test)
op32 = model32.predict(x_test)

# 그림

fig, axes = plt.subplots(7, 5, figsize = (15, 15))

random_imgs = random.sample(range(op01.shape[0]), 5)

outputs = [x_test, op01, op02, op04, op08, op16, op32]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28), cmap = 'gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()

 
