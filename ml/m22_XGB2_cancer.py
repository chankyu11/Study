from xgboost import XGBClassifier, XGBRFRegressor
from xgboost import plot_importance                 # plot이니까 그림 그리는 듯. 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ds = load_breast_cancer()

x = ds.data
y = ds.target
# print(x.shape) # (506, 13)
# print(y.shape) # (506 ,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

n_estimators = 500          # 나무의 숫자 
learning_rate = 0.01        # 학습률
colsample_bytree = 0.9      # 대개 0.6 ~ 0.9를 사용한다. 기본값은 1
colsample_bylevel = 0.9     # 샘플의 레벨을 얼마나 사용하겠냐라는 뜻. 0.6 ~ 0.9 
# XGB에서 위 4가지는 무조건 사용.
max_depth = 5               # 
n_jobs = -1                 # 
# 디시젼 트리는 트리에 들어있고, 랜포는 앙상블에 들어감.
# 트리모양에서는 결측지 제거, 전처리 불필요.

model = XGBClassifier(max_depth = max_depth,
                     learning_rate = learning_rate,
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
