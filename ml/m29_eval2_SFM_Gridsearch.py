# xgboost evaluate

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor,XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score,accuracy_score
from sklearn.datasets import load_breast_cancer

## 데이터
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    shuffle = True, random_state = 66)

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

## 모델링
model = XGBClassifier(n_estimators = 300,        # verbose의 갯수, epochs와 동일
                     learning_rate = 0.1)

model.fit(x_train, y_train,
          verbose = True, eval_metric = ['error', 'auc'],
          eval_set = [(x_train, y_train),
                      (x_test, y_test)])
        #   early_stopping_rounds = 100)
# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)

results = model.evals_result()
print("eval's result : ", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
# print("r2 Score : %.2f%%" %(r2 * 100))
print("acc : ", acc)

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
'''
acc :  0.9736842105263158
[0.00046491 0.0013246  0.00139511 0.00200115 0.00204048 0.00244829
 0.00346774 0.00359193 0.00407993 0.00500716 0.00536249 0.00552303
 0.00699229 0.00727135 0.00831567 0.00982234 0.01391234 0.01405333
 0.01454801 0.01663779 0.01722644 0.01724869 0.02118052 0.0229155
 0.03107507 0.10279346 0.11432321 0.1687874  0.16907775 0.20711204]
'''