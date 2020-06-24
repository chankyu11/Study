import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

## 데이터
boston = load_boston()
x, y = load_boston(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)


## 모델링
model = LGBMRegressor(n_estimators = 1000,         # verbose의 갯수, epochs와 동일
                     num_leaves = 50,
                     subsample = 0.8,
                     min_child_samples = 60,
                     max_depth = 20)

model.fit(x_train, y_train,
          verbose = True, eval_metric = ['rmse','logloss'],  # 리스트로 묶어서 매트릭스 두개 사용가능
          eval_set = [(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 3)  # ealrystopping 


# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)
thresholds = np.sort(model.feature_importances_)
print(thresholds)

import pickle

for thresh in thresholds:  
    # 칼럼수 만큼 돈다.
    selection = SelectFromModel(model, threshold = thresh, prefit= True)

    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape)

    selection_model = LGBMRegressor()
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(selection_x_test)

    score = r2_score(y_test, y_pred)
    print("R2: ", score)
    
 
    print("Thresh= %.3f, n = %d, R2: %.2f%%" %(thresh, selection_x_train.shape[1],
                        score * 100.0))

y_pred = model.predict(x_test)

print(model.feature_importances_)

r2 = r2_score(y_pred, y_test)
print("r2: %.2f" % (r2 * 100.0))

def plot_feature_importances_cancer(model):
    n_features = boston.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align = 'center')
    # bar는 세로 막대, barh 가로 막대
    plt.yticks(np.arange(n_features), boston.feature_names)
    # y축의 정보
    
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    # ylim = y축 범위

plot_feature_importances_cancer(model)
plt.show()