# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피처수를 줄인다.
# 3. regularization

from xgboost import XGBClassifier, XGBRFRegressor
from xgboost import plot_importance                 # plot이니까 그림 그리는 듯. 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ds = load_boston()

x = ds.data
y = ds.target
# print(x.shape) # (506, 13)
# print(y.shape) # (506 ,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

n_estimators = 1000          # 나무의 숫자 
learning_rate = 0.1        # 학습률
colsample_bytree = 0.9      # 대개 0.6 ~ 0.9를 사용한다. 기본값은 1
colsample_bylevel = 0.9     # 샘플의 레벨을 얼마나 사용하겠냐라는 뜻. 0.6 ~ 0.9 
# XGB에서 위 4가지는 무조건 사용.
max_depth = 5               # 
n_jobs = -1                 # 
# 디시젼 트리는 트리에 들어있고, 랜포는 앙상블에 들어감.
# 트리모양에서는 결측지 제거, 전처리 불필요.

model = XGBRFRegressor(max_depth = max_depth,
                    #  learning_rate = learning_rate,
                     n_estimators = n_estimators,
                     n_jobs = n_jobs,
                     colsample_bytree = colsample_bytree,
                     colsample_bylevel = colsample_bylevel)

# learning_rate를 빼니까 0.92까지는 올라감. 왜지

# model = XGBRFRegressor(max_depth = max_depth,
#                        learning_rate = learning_rate)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('Score : ', score)

print(model.feature_importances_)
# print(model.best_estimator_)
# print(model.best_params_)

plot_importance(model)
# plt.show()

""" 

n_estimators = 200 이하 동일
점수:  -5.761939920525989
==================================================
[0.03953733 0.01244145 0.06104018 0.03108958 0.06728637 0.22789137
 0.02047721 0.06836524 0.02038038 0.04713318 0.12853344 0.0158075
 0.2600167 ] 
 
 """