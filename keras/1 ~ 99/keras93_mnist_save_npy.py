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

np.save('./data/mnist_train_x.npy', arr = x_train)
np.save('./data/mnist_test_x.npy', arr = x_test)
np.save('./data/mnist_train_y.npy', arr = y_train)
np.save('./data/mnist_test_y.npy', arr = y_test)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape) # (60000, 10)
# print(y_test.shape)
# print(y_train)
# print(y_test)

# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
# x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

