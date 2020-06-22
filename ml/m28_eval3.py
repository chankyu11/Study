""" 
1. eval에 'loss'와 다른 지표 1개 더 추가.
2. earlyStopping 적용
3. plot으로 그려라. 

다중 분류
"""

import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston, load_breast_cancer, load_iris

## 데이터
x, y = load_iris(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)

## 모델링

model = XGBClassifier(objective='multi:softprob', n_estimators=100, learning_rate=0.1, n_jobs=-1)

# model = XGBClassifier(n_estimators = 100,        # verbose의 갯수, epochs와 동일
#                      learning_rate = 0.1)

model.fit(x_train, y_train,
          verbose = True, eval_metric = ['mlogloss','merror'],  # 리스트로 묶어서 매트릭스 두개 사용가능
          eval_set = [(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 200)  # ealrystopping 

# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다), merror: Multiclass classification error rate

results = model.evals_result()
# print("eval's result : ", results)

y_pred = model.predict(x_test)

# r2 = r2_score(y_pred, y_test)
# print("r2: %.2f" % (r2 * 100.0))

acc = accuracy_score(y_pred, y_test)
print("acc: %.2f" %(acc * 100.0))

# 시각화

import matplotlib.pyplot as plt
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label = 'Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label = 'Test')
ax.legend()
plt.ylabel('mlogloss')
plt.title('XGBoost mlogloss')
plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label = 'Train')
ax.plot(x_axis, results['validation_1']['merror'], label = 'Test')
ax.legend()
plt.ylabel('merror')
plt.title('XGBoost merror')
plt.show()

