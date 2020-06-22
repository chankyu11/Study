# xgboost evaluate

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor,XGBRFClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score,accuracy_score
from sklearn.datasets import load_breast_cancer

## 데이터
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)

## 모델링
model = XGBRFClassifier(n_estimators = 300,        # verbose의 갯수, epochs와 동일
                     learning_rate = 0.1)

model.fit(x_train, y_train,
          verbose = 1, eval_metric = ['error', 'auc'],
          eval_set = [(x_train, y_train),
                      (x_test, y_test)])
        #   early_stopping_rounds = 100)
# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

import pickle

for thresh in thresholds: #중요하지 않은 컬럼들을 하나씩 지워나간다.
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
   

    print(selection_x_train.shape)
    
    selection_model = XGBRFClassifier(n_jobs=-1)

    selection_model.fit(selection_x_train,y_train, eval_metric = ['error', 'auc'],
          eval_set = [(selection_x_train, y_train),
                      (selection_x_test, y_test)])

    y_pred = selection_model.predict(selection_x_test)

    acc = accuracy_score(y_test, y_pred)
    #print("R2:",r2)

    for i in thresholds:
        pickle.dump(selection_model, open("./model/xgb_save/cancel/cancel.pickle{}.dat".format(selection_x_train.shape[1]), "wb"))


    print("Thresh=%.3f, n=%d, acc: %.2f%%" %(thresh, selection_x_train.shape[1],
                        acc*100.0))


results = model.evals_result()
print("eval's result : ", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
# print("r2 Score : %.2f%%" %(r2 * 100))
print("acc : ", acc)

'''
[0]     validation_0-error:0.01099      validation_0-auc:0.99990        validation_1-error:0.02632      validation_1-auc:0.97895
Thresh=0.002, n=30, acc: 97.37%
(455, 29)
[0]     validation_0-error:0.01099      validation_0-auc:0.99992        validation_1-error:0.02632      validation_1-auc:0.97845
Thresh=0.003, n=29, acc: 97.37%
(455, 28)
[0]     validation_0-error:0.01099      validation_0-auc:0.99988        validation_1-error:0.02632      validation_1-auc:0.97895
Thresh=0.003, n=28, acc: 97.37%
(455, 27)
[0]     validation_0-error:0.01099      validation_0-auc:0.99992        validation_1-error:0.03509      validation_1-auc:0.97845
Thresh=0.003, n=27, acc: 96.49%
(455, 26)
[0]     validation_0-error:0.01099      validation_0-auc:0.99992        validation_1-error:0.02632      validation_1-auc:0.97828
Thresh=0.005, n=26, acc: 97.37%
(455, 25)
[0]     validation_0-error:0.01099      validation_0-auc:0.99992        validation_1-error:0.03509      validation_1-auc:0.97845
Thresh=0.005, n=25, acc: 96.49%
(455, 24)
[0]     validation_0-error:0.01099      validation_0-auc:0.99990        validation_1-error:0.03509      validation_1-auc:0.97912
Thresh=0.005, n=24, acc: 96.49%
(455, 23)
[0]     validation_0-error:0.01099      validation_0-auc:0.99990        validation_1-error:0.03509      validation_1-auc:0.97878
Thresh=0.006, n=23, acc: 96.49%
(455, 22)
[0]     validation_0-error:0.01099      validation_0-auc:0.99992        validation_1-error:0.03509      validation_1-auc:0.97895
Thresh=0.007, n=22, acc: 96.49%
(455, 21)
[0]     validation_0-error:0.01099      validation_0-auc:0.99990        validation_1-error:0.03509      validation_1-auc:0.97928
Thresh=0.007, n=21, acc: 96.49%
(455, 20)
[0]     validation_0-error:0.01099      validation_0-auc:0.99988        validation_1-error:0.03509      validation_1-auc:0.97812
Thresh=0.007, n=20, acc: 96.49%
(455, 19)
[0]     validation_0-error:0.01099      validation_0-auc:0.99990        validation_1-error:0.03509      validation_1-auc:0.98463
Thresh=0.008, n=19, acc: 96.49%
(455, 18)
[0]     validation_0-error:0.01099      validation_0-auc:0.99990        validation_1-error:0.03509      validation_1-auc:0.98296
Thresh=0.008, n=18, acc: 96.49%
(455, 17)
[0]     validation_0-error:0.01099      validation_0-auc:0.99986        validation_1-error:0.03509      validation_1-auc:0.98613
Thresh=0.009, n=17, acc: 96.49%
(455, 16)
[0]     validation_0-error:0.01099      validation_0-auc:0.99988        validation_1-error:0.03509      validation_1-auc:0.97895
Thresh=0.009, n=16, acc: 96.49%
(455, 15)
[0]     validation_0-error:0.01099      validation_0-auc:0.99984        validation_1-error:0.03509      validation_1-auc:0.97912
Thresh=0.010, n=15, acc: 96.49%
(455, 14)
[0]     validation_0-error:0.01099      validation_0-auc:0.99988        validation_1-error:0.03509      validation_1-auc:0.98480
Thresh=0.010, n=14, acc: 96.49%
(455, 13)
[0]     validation_0-error:0.01099      validation_0-auc:0.99982        validation_1-error:0.03509      validation_1-auc:0.98513
Thresh=0.010, n=13, acc: 96.49%
(455, 12)
[0]     validation_0-error:0.01099      validation_0-auc:0.99986        validation_1-error:0.03509      validation_1-auc:0.97828
Thresh=0.011, n=12, acc: 96.49%
(455, 11)
[0]     validation_0-error:0.01099      validation_0-auc:0.99982        validation_1-error:0.03509      validation_1-auc:0.98597
Thresh=0.011, n=11, acc: 96.49%
(455, 10)
[0]     validation_0-error:0.01099      validation_0-auc:0.99975        validation_1-error:0.03509      validation_1-auc:0.98162
Thresh=0.012, n=10, acc: 96.49%
(455, 9)
[0]     validation_0-error:0.01099      validation_0-auc:0.99926        validation_1-error:0.03509      validation_1-auc:0.97711
Thresh=0.012, n=9, acc: 96.49%
(455, 8)
[0]     validation_0-error:0.01099      validation_0-auc:0.99936        validation_1-error:0.03509      validation_1-auc:0.98079
Thresh=0.013, n=8, acc: 96.49%
(455, 7)
[0]     validation_0-error:0.01099      validation_0-auc:0.99930        validation_1-error:0.03509      validation_1-auc:0.98213
Thresh=0.019, n=7, acc: 96.49%
(455, 6)
[0]     validation_0-error:0.02418      validation_0-auc:0.99808        validation_1-error:0.03509      validation_1-auc:0.97928
Thresh=0.020, n=6, acc: 96.49%
(455, 5)
[0]     validation_0-error:0.02637      validation_0-auc:0.99806        validation_1-error:0.03509      validation_1-auc:0.98079
Thresh=0.048, n=5, acc: 96.49%
(455, 4)
[0]     validation_0-error:0.03077      validation_0-auc:0.99658        validation_1-error:0.04386      validation_1-auc:0.98045
Thresh=0.096, n=4, acc: 95.61%
(455, 3)
[0]     validation_0-error:0.03736      validation_0-auc:0.99362        validation_1-error:0.08772      validation_1-auc:0.96993
Thresh=0.144, n=3, acc: 91.23%
(455, 2)
[0]     validation_0-error:0.04835      validation_0-auc:0.99222        validation_1-error:0.09649      validation_1-auc:0.97611
Thresh=0.228, n=2, acc: 90.35%
(455, 1)
[0]     validation_0-error:0.06374      validation_0-auc:0.98954        validation_1-error:0.09649      validation_1-auc:0.96542
Thresh=0.269, n=1, acc: 90.35%
eval's result :  {'validation_0': {'error': [0.010989], 'auc': [0.999918]}, 'validation_1': {'error': [0.035088], 'auc': [0.977781]}}
acc :  0.9649122807017544
'''

