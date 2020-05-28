import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train[0])
print('y_train:', y_train[0])

print(x_test.shape)  # 10000, 28, 28
print(x_train.shape) # 60000, 28, 28 
print(y_test.shape)  # 10000, 
print(y_train.shape) # 60000, 
print(x_train[0].shape) # 28, 28

plt.imshow(x_train[0], 'gray')
plt.show()
