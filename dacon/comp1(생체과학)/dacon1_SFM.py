import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
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
# print(test.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)

model = XGBRegressor(n_estimators = 100,        # verbose의 갯수, epochs와 동일
                     learning_rate = 0.1)

model.fit(x_train, y_train,
          verbose = True, eval_metric = ['error','rmse'],  # 리스트로 묶어서 매트릭스 두개 사용가능
          eval_set = [(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 3)  # ealrystopping 

thresholds = np.sort(model.feature_importances_)
print(thresholds)


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
    
    print("Thresh= %.3f, n = %d, R2: %.2f%%" %(thresh, selection_x_train.shape[1],
                        score * 100.0))

results = model.evals_result()
# print("eval's result : ", results)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print("r2: %.2f" % (r2 * 100.0))