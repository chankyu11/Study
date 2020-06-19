# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피처수를 줄인다.
# 3. regularization

from xgboost import XGBClassifier, XGBRFRegressor
from xgboost import plot_importance                 # plot이니까 그림 그리는 듯. 
from sklearn.datasets import load_iris, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# ds = load_iris()
# ds = load_boston()
ds = load_breast_cancer()

x = ds.data
y = ds.target

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

param = [{"n_estimators": [100, 200 ,300], "learning_rate": [0.1, 0.2, 0.5, 0.01],
          "max_depth": [4,5,6]},
         {"n_estimators": [90, 100 ,110], "learning_rate": [0.1, 0.001, 0.01],
          "max_depth": [4,5,6], "colsample_bytree": [0.6, 0.9, 1]},
         {"n_estimators": [90, 110], "learning_rate": [0.1, 0.001, 0.075],
          "max_depth": [4,5,6], "colsample_bytree": [0.6, 0.9, 1], "colsample_bylevel": [0.6, 0.7, 0.9]}
]

model = GridSearchCV(XGBClassifier(), param, cv = 5, n_jobs = -1)

model.fit(x_train, y_train)

print("=" * 50)
print(model.best_estimator_)
print(model.best_params_)
print("=" * 50)
score = model.score(x_test, y_test)
print('Score : ', score)
''' 
iris
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.001, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=90, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
{'colsample_bytree': 0.6, 'learning_rate': 0.001, 'max_depth': 4, 'n_estimators': 90}
==================================================
Score :  0.9666666666666667 
'''