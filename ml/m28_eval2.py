""" 
1. eval에 'loss'와 다른 지표 1개 더 추가.
2. earlyStopping 적용
3. plot으로 그려라. 

이진 분류
"""

# xgboost evaluate

import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston, load_breast_cancer

## 데이터
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)

## 모델링
model = XGBClassifier(n_estimators = 100,        # verbose의 갯수, epochs와 동일
                     learning_rate = 0.1)

model.fit(x_train, y_train,
          verbose = True, eval_metric = ['auc','error'],  # 리스트로 묶어서 매트릭스 두개 사용가능
          eval_set = [(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 20)  # ealrystopping 

# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)

results = model.evals_result()
print("eval's result : ", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_pred, y_test)
print("acc: %.2f" %(acc * 100.0))

# 시각화

import matplotlib.pyplot as plt
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label = 'Train')
ax.plot(x_axis, results['validation_1']['auc'], label = 'Test')
ax.legend()
plt.ylabel('auc')
plt.title('XGBoost auc')
plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label = 'Train')
ax.plot(x_axis, results['validation_1']['error'], label = 'Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBoost error')
plt.show()