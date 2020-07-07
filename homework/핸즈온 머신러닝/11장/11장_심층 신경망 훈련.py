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


# p.440
# RMSProp 케라스에 옵티마이저 존재
optimizer = keras.optimizers.RMSProp(lr = 0.001, rho = 0.9)
# lr = 0.001, rho = 0.9가 기본값
# 아주 간단한 문제를 제외하고 RMSProp가 AdaGrad보다 좋음 

# p.441
# Adam 만드는 방법
optimizer = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.9)


# p.443
# 옵티마이저 비교
'''
클래스                                  수렴속도            수렴품질
SGD                                        *                  ***
SGD(momentum = ...)                        **                 ***
SGD(momentum = ..., nesterov = True)       **                 ***
Adagrad                                    ***                *(너무 일찍 멈춤)
RMSprop                                    ***                ** 또는 ***
Adam                                       ***                ** 또는 ***
Nadam                                      ***                ** 또는 ***
AdaMax                                     ***                ** 또는 ***

* = 나쁨, ** = 보통, *** = 좋음
'''

# p.444
# 학습률 스케쥴링

# 학습률 스케쥴링은 훈련하는 동안 학습률을 감소시키는 전략

# p.446
# 케라스에서 거듭제곱 기반 스케쥴링이 가장 구현이 쉬움, 옵티마이저를 만들때 decay 매개변수만 지정하면 끝
optimizer = keras.optimizers.SGD(lr = 0.01, dacay = 1e-4)

# decay는 학습률을 나누기 위해 수행항 스텝수

# 지수 기반 스케쥴링: 에포크를 받아 학습률을 반환하는 함수를 정의하면 끝 

def exponential_dacay_fn(epoch):
    return 0.01 * 0.1**(epoch / 20)

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_dacay_fn
exponential_dacay_fn = exponential_decay(lr0=0.01, s =20)

# 함수를 이렇게 정의한 후 LearningRateScheduler 콜백을 만들고 이 콜백을 fit()에 전달.

lr_scheduler = keras.callbacks.LeaningRateScheduler(exponential_dacay_fn)
history = model.fit(x_train_scaled, y_train,callbacks=[lr_scheduler])

def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
        
#최상의 검증 손실이 다섯 번의 연속적인 에포크 동안 향상되지 않을 떄마다 학습률에 0.5*
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

s = 20 * len(x_train) //32
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)

# 규제를 사용해 과대적합 피하기
# L1, L2 규제, Dropout, Max-norm 이 있음.

from functools import partial
layer = keras.layers.Dense(100, activation='elu',
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))

# python의 partial() 함수를 사용하여 기본 매개변수 값을 사용해 함수 호출을 감쌈.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28],
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation='softmax',
                        kernel_initializer='glorot_uniform')
])

# p.453

# 드롭아웃

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation='softmax')])
    
#몬테 카를로 드롭아웃 (훈련된 모델을 재훈련하거나 수정하지않고 성능을 크게향상시킬수있다.)
y_probas = np.stack([model(x_test_scaled, training=True)
                    for sample in range(100)])
y_proba = y_probas.mean(axis=0)


#배치 정규화와 같은 층을 가지고 있다면 dropout층을 다음과 같이 변경
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

#맥스-노름 규제
keras.layers.Dense(100, activation='elu', kernel_initializer="he_normal",
                    kernel_constraint=keras.constraints.max_norm(1.))
