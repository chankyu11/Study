from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

ds = load_boston()

x = ds.data
y = ds.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

model = XGBRFRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2: " , score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

import time
# (시작 시간 - 종료시간)사용된 시간을 알아볼 수 있음

start = time.time()

for thresh in thresholds:  
    # 칼럼수 만큼 돈다.
    selection = SelectFromModel(model, threshold = thresh, prefit= True)

    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape)

    selection_model = XGBRFRegressor(n_estimator = 500)
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(selection_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2: ", score)

    print("Thresh= %.3f, n = %d, R2: %.2f%%" %(thresh, selection_x_train.shape[1],
                        score * 100.0))

end = time.time() - start

print("소요 시간:", end)
print("====" * 20)

start2 = time.time()

for thresh in thresholds:  
    # 칼럼수 만큼 돈다.
    selection = SelectFromModel(model, threshold = thresh, prefit= True)

    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape)

    selection_model = XGBRFRegressor(n_jobs = -1, n_estimator = 500)
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(selection_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2: ", score)

    print("Thresh= %.3f, n = %d, R2: %.2f%%" %(thresh, selection_x_train.shape[1],
                        score * 100.0))
end2 = time.time() - start2

print("소요 시간:", end2)
