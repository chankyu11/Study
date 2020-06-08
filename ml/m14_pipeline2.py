import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# 1. 데이터
ds = load_iris()

x = ds.data
y = ds.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                    train_size = 0.8, 
                                    shuffle = True, random_state = 256)

# 그리드/ 랜덤 서치에서 사용할 매개 변수
parameter = [
        {"svm__C": [1,10,100,1000], "svm__kernel": ['linear']},
        {"svm__C": [1,10,100,1000], "svm__kernel": ['rbf'], 'svm__gamma': [0.001, 0.0001]},
        {"svm__C": [1,10,100,1000], "svm__kernel": ['sigmoid'], 'svm__gamma': [0.001, 0.0001]}
]

# param = [
#     {"C" : [1,10,100,1000],"kernel": ["linear","rbf","sigmoid"]},
#     {"C" : [1,10,100,1000],"kernel": ["rbf"],"gamma" : [0.001,0.0001]},
#     {"C" : [1,10,100,1000],"kernel": ["sigmoid" ],"gamma" : [0.001,0.0001]}
# ]

# 2. 모델

# model = SVC()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

model = RandomizedSearchCV(pipe, parameter, cv = 5)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print("최적의 매개변수 : ", model.best_estimator_)
print("acc:", acc)

import sklearn as sk
print("sklearn: ", sk.__version__)