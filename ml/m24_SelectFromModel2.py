from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV

x, y = load_breast_cancer(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)


param = [{'n_estimators': [100,150,200,250,300],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'max_depth': [4,5,6]},
    {'n_estimators': [100,150,200,250,300],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'colsample_bytree':[0.6,0.68,0.9,1],
    'max_depth': [4,5,6]},
    {'n_estimators': [100,150,200,250,300],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'colsample_bylevel': [0.6,0.68,0.9,1],
    'max_depth': [4,5,6]}
    ]


model = XGBClassifier(n_jobs = -1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2: " , score)

thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds:  
    # 칼럼수 만큼 돈다.
    selection = SelectFromModel(model, threshold = thresh, prefit= True)

    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape)

    selection_model = GridSearchCV(XGBClassifier(), param, n_jobs = -1, cv = 5)
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(selection_x_test)

    score = accuracy_score(y_test, y_pred)
    # print("R2: ", score)

    print("Thresh= %.3f, n = %d, Acc: %.2f%%" %(thresh, selection_x_train.shape[1], score * 100.0))