import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
import pickle as pk

## 데이터
x, y = load_boston(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)


## 모델링
model = XGBRegressor(n_estimators = 100,        # verbose의 갯수, epochs와 동일
                     learning_rate = 0.1)

model.fit(x_train, y_train,
          verbose = True, eval_metric = ['error','rmse'],  # 리스트로 묶어서 매트릭스 두개 사용가능
          eval_set = [(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 3)  # ealrystopping 

# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)
thresholds = np.sort(model.feature_importances_)
print(thresholds)

import pickle as pk

for thresh in thresholds:  
    # 칼럼수 만큼 돈다.
    selection = SelectFromModel(model, threshold = thresh, prefit= True)

    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape)

    selection_model = XGBRegressor(n_jobs = -1)
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(selection_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2: ", score)
    
    for i in thresholds:
        pk.dump(selection_model, open("./model/xgb_save/boston.pickle{}.dat".format(selection_x_train.shape[1]), "wb"))

    print("Thresh= %.3f, n = %d, R2: %.2f%%" %(thresh, selection_x_train.shape[1],
                        score * 100.0))

results = model.evals_result()
# print("eval's result : ", results)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print("r2: %.2f" % (r2 * 100.0))

# 불러오기
model2 = pk.load(open("./model/xgb_save/boston/boston.pickle1.dat","rb"))
print("불러옴")
y_pred = model2.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("r2: %.2f" % (r2 * 100.0))



# 모델 불러오기
# 1.  데이터
x, y = load_boston(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)

model2 = XGBRegressor()
model2.fit(x_train, y_train,
          verbose = True, eval_metric = ['error','rmse'],  # 리스트로 묶어서 매트릭스 두개 사용가능
          eval_set = [(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 100)  # ealrystopping 
model2 = pk.load(open("./model/xgb_save/boston/boston.pickle1.dat","rb"))
print("불러옴")

y_pred = model2.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("r2: %.2f" % (r2 * 100.0))
