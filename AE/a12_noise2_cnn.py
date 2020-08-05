from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.datasets import mnist
import numpy as np

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size * 2, kernel_size=(3,3), padding='valid', input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add(Conv2D(filters=hidden_layer_size * 1, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(UpSampling2D(size=(3,3)))

    model.add(Conv2D(filters=hidden_layer_size * 2, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(UpSampling2D(size=(3,3)))

    model.add(Conv2D(filters=hidden_layer_size * 3, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid'))
    
    model.summary()
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()

x_train, y_train = train_set
x_test, y_test = test_set

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) / 255.
# print(x_train.shape)
# print(x_test.shape)

# noise
x_train_noised = x_train + np.random.normal(0, 0.4, size = x_train.shape)
# random.normal 정규분포에 따른, 0을 평균 0.5 표준편차
x_test_noised = x_test + np.random.normal(0, 0.4, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)

model = autoencoder(hidden_layer_size = 32)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

model.fit(x_train_noised, x_train, epochs = 10)

op = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
 
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(op.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("INPUT", size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈!
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("noise", size = 30)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    

# 오도인코더가 출력한 이미지를 아래에 그린다.

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(op[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
