from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))

'''
사용자 정의 콜백은 훈련하는 동안 검증 손실과 훈련 손실의 비율을 출력

on_train_begin(), on_train_end(), on_epoch_begin(), on_epoch_end()
on_batch_begin(), on_batch_end()를 구현 가능.
검증단계에서 on_test_begin()
예측단계에서 on_predict_begin()
'''