# cifar10으로 autoencoder 구성!

# DNN

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10

# 모델 함수
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, input_shape = (3072, ), activation = 'relu'))
    model.add(Dense(units = hidden_layer_size, input_shape = (3072, ), activation = 'relu'))
    model.add(Dense(units = 3072, activation = 'sigmoid'))
    return model

# 데이터

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape)   # (50000, 32, 32, 3)
# print(y_train.shape)   # (50000, 1)
# print(x_test.shape)    # (10000, 32, 32, 3)
# print(y_test.shape)    # (10000, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * 3) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * 3) / 255.

model = autoencoder(hidden_layer_size = 1000)


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

model.fit(x_train, x_train, epochs = 20, validation_split = 0.1)

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
