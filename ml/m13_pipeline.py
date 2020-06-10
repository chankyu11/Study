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

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
pipe = make_pipeline(MinMaxScaler(), SVC())
# pipeline은 scaler 쓰고 어떤 기법을 쓸지 명시, 모델쓰고, 기법 명시
# make pipeline은 (전처리, 모델)

pipe.fit(x_train, y_train)
print("acc :", pipe.score(x_test, y_test))

