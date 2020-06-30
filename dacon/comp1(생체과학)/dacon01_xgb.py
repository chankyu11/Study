import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRFRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

train = np.load('D:/STUDY/dacon/comp1/train_bfill.npy')
test = np.load('D:/STUDY/dacon/comp1/test_bfill.npy')
submission = np.load('D:/STUDY/dacon/comp1/sample_submission_bfill.npy')

x = train[:, :71]
y = train[:, 71:]
test = test[:, :71]
# print(x.shape)         # 10000, 71

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle='True', random_state=1)
# print(x_train.shape)   # 8000, 71

param = [{"n_estimators": [100, 200 ,300], "learning_rate": [0.1, 0.2, 0.5, 0.01],
          "max_depth": [4,5,6]},
         {"n_estimators": [90, 100 ,110], "learning_rate": [0.1, 0.001, 0.01],
          "max_depth": [4,5,6], "colsample_bytree": [0.6, 0.9, 1]},
         {"n_estimators": [90, 110], "learning_rate": [0.1, 0.001, 0.075],
          "max_depth": [4,5,6], "colsample_bytree": [0.6, 0.9, 1], "colsample_bylevel": [0.6, 0.7, 0.9]}
]
model = XGBRFRegressor(n_jobs=-1)
model = GridSearchCV(model, param, cv =5)
model1 = MultiOutputRegressor(model)

warnings.filterwarnings('ignore')
model1.fit(x_train, y_train)

y_pred = model1.predict(test)
print(y_pred)
print(y_pred.shape)

# mae = mean_absolute_error(y_test, y_pred)
# warnings.filterwarnings('ignore')
# print(mae)

score =r2_score(y_test, y_pred)
print(score)

# print("최적의 매개 변수 :  ", model.best_params_)
# warnings.filterwarnings('ignore')
# print("최적의모델은:", model.best_estimator_)

# a = np.arange(10000,20000)
# y_pred = pd.DataFrame(y_pred,a)
# y_pred.to_csv('D:/STUDY/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

print("완료")
'''
100 0.01 0.9 0.9 10
[[0.5504444  0.53717685 0.59504706 0.5260649 ]
 [0.5977015  0.53983396 0.5888328  0.5251515 ]
 [0.58779496 0.5380208  0.57877254 0.5208966 ]
 ...
 [0.5918929  0.5372582  0.5858242  0.5248908 ]
 [0.5930071  0.53776    0.60379905 0.5214697 ]
 [0.5824901  0.5337728  0.5854641  0.53400016]]
-7.128554596553316
'''