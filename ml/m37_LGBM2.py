import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston, load_breast_cancer
from lightgbm import LGBMRegressor, LGBMClassifier

## 데이터
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)


## 모델링
model = LGBMClassifier(n_estimators = 1000,         # verbose의 갯수, epochs와 동일
                     num_leaves = 50,
                     subsample = 0.8,
                     min_child_samples = 60,
                     max_depth = -1)

model.fit(x_train, y_train,
          verbose = True, eval_metric = ['auc','error'],  # 리스트로 묶어서 매트릭스 두개 사용가능
          eval_set = [(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 30)  # ealrystopping 


# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)
thresholds = np.sort(model.feature_importances_)
print(thresholds)

import pickle

for thresh in thresholds:  
    # 칼럼수 만큼 돈다.
    selection = SelectFromModel(model, threshold = thresh, prefit= True)

    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape)

    selection_model = LGBMClassifier(n_jobs = -1)
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(selection_x_test)

    score = accuracy_score(y_test, y_pred)
    print("ACC: ", score)
    # score = r2_score(y_test, y_pred)
    # print("R2: ", score)
    
 
    print("Thresh= %.3f, n = %d, R2: %.2f%%" %(thresh, selection_x_train.shape[1],
                        score * 100.0))

y_pred = model.predict(x_test)

score = accuracy_score(y_test, y_pred)
print("ACC: ", score)