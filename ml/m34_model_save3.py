
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

thresholds = np.sort(model.feature_importances_)
print(thresholds)

import pickle as pk

for thresh in thresholds: #중요하지 않은 컬럼들을 하나씩 지워나간다.
    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
   

    print(selection_x_train.shape)
    
    selection_model = XGBClassifier(n_jobs=-1)

    selection_model.fit(selection_x_train,y_train, eval_metric = ['mlogloss', 'merror'],
          eval_set = [(selection_x_train, y_train),
                      (selection_x_test, y_test)])

    y_pred = selection_model.predict(selection_x_test)
        
    acc = accuracy_score(y_test, y_pred)
    #print("R2:",r2)
    for thresh in thresholds:
        pk.dump(selection_model, open("./model/xgb_save/iris/iris.pickle{}.dat".format(selection_x_train.shape[1]), "wb"))
        # for 문으로 돌면서 매번 저장.
    print("Thresh=%.3f, n=%d, acc: %.2f%%" %(thresh, selection_x_train.shape[1],
                        acc*100.0))


results = model.evals_result()
# print("eval's result : ", results)

y_pred = model.predict(x_test)

# r2 = r2_score(y_pred, y_test)
# print("r2: %.2f" % (r2 * 100.0))

acc = accuracy_score(y_pred, y_test)
print("acc: %.2f" %(acc * 100.0))


# 불러오기
model2 = pk.load(open("./model/xgb_save/iris/iris.pickle1.dat","rb"))
print("불러옴")
y_pred = model2.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("r2: %.2f" % (r2 * 100.0))