from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.datasets import mnist
import numpy as np

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size * 4, kernel_size=(3,3), padding='valid', input_shape=(32,32,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(filters=hidden_layer_size * 2, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(filters = hidden_layer_size, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(Conv2D(filters = 10, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(Conv2D(filters = 3, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(UpSampling2D(size=(2,2)))
    
    model.summary()
    return model

from tensorflow.keras.datasets import cifar10

train_set, test_set = cifar10.load_data()

x_train, y_train = train_set
x_test, y_test = test_set

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3) / 255.
# print(x_train.shape)
# print(x_test.shape)

model = autoencoder(hidden_layer_size = 32)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

model.fit(x_train, x_train, epochs = 10)

op = model.predict(x_test)

from matplotlib import pyplot as plt
import random
 
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize = (10, 7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(op.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(32,32, 3), cmap = 'coolwarm')
    if i == 0:
        ax.set_ylabel("INPUT", size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오도인코더가 출력한 이미지를 아래에 그린다.

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(op[random_images[i]].reshape(32, 32, 3), cmap = 'coolwarm')
    if i == 0:
        ax.set_ylabel("OUTPUT", size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()