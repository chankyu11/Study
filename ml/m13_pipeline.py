import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# 1. 데이터
ds = load_iris()

x = ds.data
y = ds.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                    train_size = 0.8, 
                                    shuffle = True, random_state = 256)

# 2. 모델

# model = SVC()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

pipe.fit(x_train, y_train)
print("acc :", pipe.score(x_test, y_test))

