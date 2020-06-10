import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# 1. 데이터
ds = load_iris()

x = ds.data
y = ds.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                    train_size = 0.8, 
                                    shuffle = True, random_state = 256)

# 그리드/ 랜덤 서치에서 사용할 매개 변수

# parameter = [
#         {"svc__C": [1,10,100,1000], "svm__kernel": ['linear']},
#         {"svc__C": [1,10,100,1000], "svm__kernel": ['rbf'], 'svm__gamma': [0.001, 0.0001]},
#         {"svc__C": [1,10,100,1000], "svm__kernel": ['sigmoid'], 'svm__gamma': [0.001, 0.0001]}
# ]

parameter = [
        {"rf__n_estimators": [2, 5, 10]},
        {"rf__max_depth": [1,10,100,1000]},
        {"rf__min_samples_leaf": [1,10,100,1000]}
]
#       名__决定因素
# 
# param = [
#     {"C" : [1,10,100,1000],"kernel": ["linear","rbf","sigmoid"]},
#     {"C" : [1,10,100,1000],"kernel": ["rbf"],"gamma" : [0.001,0.0001]},
#     {"C" : [1,10,100,1000],"kernel": ["sigmoid" ],"gamma" : [0.001,0.0001]}
# ]

# 2. 모델

# model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ('rf', RandomForestClassifier())])
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

model = RandomizedSearchCV(pipe, parameter, cv = 5)

# 3. 훈련

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print("최적의 매개변수 : ", model.best_estimator_)
# 변수들
print("최고의 매개변수 : ", model.best_params_)
# 최고의 매개변수
print("acc:", acc)



import sklearn as sk
print("sklearn: ", sk.__version__)