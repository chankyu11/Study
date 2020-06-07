import pandas as pd
from sklearn.model_selection import train_test_split ,cross_val_score, KFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.datasets import mnist
from sklearn.datasets import load_breast_cancer

# RandomizedSearchCV
# 1. 데이터

breast_cancer = load_breast_cancer()

x = breast_cancer.data
y = breast_cancer.target

# print(x.shape) # (569, 30)
# print(y.shape) # (569, )

# print(x_train.shape)    # (60000, 28, 28)
# print(x_test.shape)     # (10000, 28, 28)
# print(y_train.shape)    # (60000, )
# print(y_test.shape)     # (10000, )

# iris = pd.read_csv('./data/CSV/iris.csv', header = 0)

# x = iris.iloc[:, 0:4]
# y = iris.iloc[:, 4]
# # print(x.shape)
# # print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)

param ={ 'n_estimators' : [10, 50],
           'max_depth' : [10, 12, 14],
           'min_samples_leaf' : [10, 20, 30],
           'min_samples_split' : [2, 4, 20]}

kfold = KFold(n_splits =5, shuffle = True)
# # 5등분으로 나눠라!!

model = RandomizedSearchCV(RandomForestClassifier(), param, cv = kfold, n_jobs = -1)

model.fit(x_train, y_train)

print("최적의 매개변수: ", model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률: ", accuracy_score(y_test, y_pred))

'''
최적의 매개변수:  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=10, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
최종 정답률:  0.9649122807017544
'''