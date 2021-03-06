
import pandas as pd
from sklearn.model_selection import train_test_split ,cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC

# GridSearchCV
# 1. 데이터
iris = pd.read_csv('./data/CSV/iris.csv', header = 0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]
# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)

param = [{"C": [1,10,100,1000], "kernel":["linear"]},
         {"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
         {"C": [1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}]

kfold = KFold(n_splits =5, shuffle = True)
# 5등분으로 나눠라!!

model = GridSearchCV(SVC(), param, cv = kfold)

model.fit(x_train, y_train)

print("최적의 매개변수: ", model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률: ", accuracy_score(y_test, y_pred))

'''
최적의 매개변수:  SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
최종 정답률:  0.9666666666666667
'''