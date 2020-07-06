## 케라스에서 기본적으로 균등분포의 글로럿 초기화를 사용.
# kernel_initalizer = "he_uniform" 이나 
# kernel_initalizer = "he_normal" 로 바꿔서 He 초기화
# 소스코드 주소
# https://github.com/rickiepark/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# p.415
# 글로럿초기화

# keras.layers.Dense(10, activation = 'relu', kernel_initalizer = "he_normal")

# ## fan(in) 대신 fan(out) 기반의 균등분포 He 초기화를 사용하고 싶다면
# ## 다음과 같이 Variance Scaling을 사용.

# he_avg_init = keras.initializers.VarianceScaling(scale = 2.,
#                                                 mode = 'fan_avg',
#                                                 distribution = 'uniform')

# keras.layers.Dense(10, activation = "sigmoid", kernel_initializer = he_avg_init)




[name for name in dir(keras.initializers) if not name.startswith("_")]
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")

init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg',
                                          distribution='uniform')
keras.layers.Dense(10, activation="relu", kernel_initializer=init)

# p.419
#LeakyRelu
model = keras.models.Sequential([
    keras.layers.Dense(10, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2)])
    # PRelu를 사용하려면 leakyRelu를 PRelu로 변경


#selu 사용시
layer =keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")

# p.423
#배치 정규화 
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer ="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer ="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

#첫 번째 배치 정규화 층의 파라미터 살펴보기
[(var.name, var.trainable) for var in model.layers[1].variables]

# 활성화 함수 이후보다 활성화 함수 이전에 배치 정규화 층을 추가하는 것이 좋다고 함.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer ="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(100, activation="elu", kernel_initializer ="he_normal",  use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(10, activation="softmax")
])

# 파라미터중 axis가 중요! 정규화할 축을 결정


# p.427
# 그레디언트 클리핑: 역전파될 때 일정 임곗값을 넘어서지 못하게 그레이디언트를 잘라내는 것

optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss='mse', optimizer=optimizer)

# 클립 벨류는 모든 원소를 -1.0 ~ 1.0사이로 클리핑한다.


# p.430
#  keras 전이학습

model_A = keras.models.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set.weights(model_A.get_weights())

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])


# p.436
# 케라스에서 모멘텀 최적화. SGD 옵티마이저를 사용 momentum에 매개변수 저장 끝

optimizer = keras.optimizers.SGD(lr = 0.001, momentum = 0.9)

