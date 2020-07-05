## P.371

import tensorflow as tf
from tensorflow import keras
# print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape)    # (60000, 28, 28)
print(X_train_full.dtype)    # ('utnt8')

x_valid, x_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000] / 255.0, y_train_full[5000:] / 255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

'''
위와 똑같은 모델
# 이런식으로 Sequential 괄호안에 작성할 수 있음
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
'''
model.summary()
# print(model.layers)
'''
<tensorflow.python.keras.layers.core.Flatten object at 0x000002AE22927B08>, 
<tensorflow.python.keras.layers.core.Dense object at 0x000002AE40FEEF48>, 
<tensorflow.python.keras.layers.core.Dense object at 0x000002AE4102DF88>, 
<tensorflow.python.keras.layers.core.Dense object at 0x000002AE41033188>]
'''
hidden1 = model.layers[1]
# print(hidden1.name)  # dense

model.get_layer(hidden1.name) is hidden1
weights, biases = hidden1.get_weights()

# print(weights)
'''
[[-0.06779739 -0.02753784 -0.01988037 ...  0.07231829  0.07297547
   0.04324508]
 [-0.04073468 -0.0553381   0.02040146 ... -0.00068119 -0.03978747
  -0.06940365]
 [ 0.00293589 -0.02944232 -0.00999056 ... -0.02279929 -0.06112693
  -0.01740805]
 ...
 [ 0.00731577  0.05559011  0.01800368 ...  0.00984228 -0.04416846
   0.0737635 ]
 [ 0.06543067  0.0057402  -0.05250499 ...  0.00361948  0.07519971
  -0.05903099]
 [ 0.03911801  0.01974247 -0.01626137 ... -0.05895227  0.00450378
   0.06175777]]
'''

# print(weights.shape)   # (784, 300)
# print(biases)
'''
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0.]
'''


model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])

'''
위와 같음.`
model.compile(loss = keras.losses.sparse_categorical_crossentropy,
              optimizer = keras.optimizers.SGD(),
              metrics = [keras.metrics.sparse_categorical_accuracy])
'''
history = model.fit(x_train, y_train, epochs=30,
                    validation_data=(x_valid, y_valid))
history.params
# print(history.epoch)
history.history.keys()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
# plt.show()

model.evaluate(x_test, y_test)

x_new = x_test[:3]
y_proba = model.predict(x_new)
print(y_proba.round(2))

# predict_classes는 확률이 가장 놓은 클래서에만 관심이 있을시 사용.
y_pred = model.predict_classes(x_new)
print(y_pred)
print(np.array(class_names)[y_pred])

