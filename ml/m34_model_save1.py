import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston

## 데이터
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
model2.load_model("./model/xgb_save/boston.pickle1.dat")
print("불러옴")

y_pred = model2.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("r2: %.2f" % (r2 * 100.0))
