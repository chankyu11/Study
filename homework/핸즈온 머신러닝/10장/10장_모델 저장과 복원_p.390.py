from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])
# 모델 저장, 불러오기
model.save("my_keras_model.h5")
model = keras.models.load_model("my_keras_model.h5")

# 가중치 저장, 불러오기
model.save_weights("my_keras_weights.ckpt")
model.load_weights("my_keras_weights.ckpt")
