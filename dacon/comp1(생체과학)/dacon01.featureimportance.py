import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('./dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./dacon/comp1/test.csv', header = 0, index_col = 0)
sm = pd.read_csv('./dacon/comp1/sample_submission.csv', header = 0, index_col = 0)

train = train.interpolate()                       
test = test.interpolate()

x_data = train.iloc[:, :71]                           
y_data = train.iloc[:, -4:]


x_data = x_data.fillna(x_data.mean())
test = test.fillna(test.mean())


x = x_data.values
y = y_data.values
x_pred = test.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state =33)

# model = DecisionTreeRegressor(max_depth =4)                     # max_depth 몇 이상 올라가면 구분 잘 못함
# model = RandomForestRegressor(max_depth=4, n_estimators= 100)
# model = GradientBoostingRegressor()
model = XGBRegressor()

def tree_fit(y_train, y_test):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print('score: ', score)
    y_predict = model.predict(x_pred)
    y_pred1 = model.predict(x_test)
    print('mae: ', mean_absolute_error(y_test, y_pred1))
    return y_predict

def boost_fit_acc(y_train, y_test):
    y_predict = []
    for i in range(len(sm.columns)):
       print(i)
       y_train1 = y_train[:, i]  
       model.fit(x_train, y_train1)
       
       y_test1 = y_test[:, i]
       score = model.score(x_test, y_test1)
       print('score: ', score)

       y_pred = model.predict(x_pred)
       y_pred1 = model.predict(x_test)
       print('mae: ', mean_absolute_error(y_test1, y_pred1))

       y_predict.append(y_pred)     
    return np.array(y_predict)

# y_predict = tree_fit(y_train, y_test)
y_predict = boost_fit_acc(y_train, y_test).reshape(-1, 4) 

# print(y_predict.shape) # (10000, 4)


# submission
a = np.arange(10000,20000)
submission = pd.DataFrame(y_predict, a)
submission.to_csv('./dacon/comp1/sub_XGB.csv',index = True, header=['hhb','hbo2','ca','na'],index_label='id')

print(model.feature_importances_)


## feature_importances
def plot_feature_importances(model):
    plt.figure(figsize= (10, 40))
    n_features = x_data.shape[1]                                # n_features = column개수 
    plt.barh(np.arange(n_features), model.feature_importances_,      # barh : 가로방향 bar chart
              align = 'center')                                      # align : 정렬 / 'edge' : x축 label이 막대 왼쪽 가장자리에 위치
    plt.yticks(np.arange(n_features), x_data.columns)          # tick = 축상의 위치표시 지점
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)             # y축의 최솟값, 최댓값을 지정/ x는 xlim

plot_feature_importances(model)
plt.show()

'''
XGboost
score:  0.05151398280090491
mae:  1.4631492877960204
(10000, 4)
[0.00714453 0.00456563 0.00523687 0.00760908 0.00678088 0.00609987
 0.01076802 0.00948863 0.01044757 0.01051742 0.00870712 0.01005827
 0.01080324 0.01445976 0.00863465 0.00999479 0.01383694 0.01053726
 0.01033302 0.013858   0.01236561 0.01186646 0.0114035  0.01210822
 0.01168458 0.01149829 0.01143724 0.01593832 0.01389337 0.01312196
 0.01431437 0.01188886 0.01445564 0.01280423 0.01480111 0.01269285
 0.01433546 0.01052709 0.01429052 0.0100729  0.01207726 0.01429687
 0.01217189 0.01429322 0.02461783 0.02595455 0.01788696 0.01153878
 0.0147355  0.01482237 0.01462235 0.01609324 0.01367005 0.01539843
 0.01336798 0.0178892  0.02250576 0.02116024 0.02215998 0.01577337
 0.02519553 0.02593758 0.03072716 0.0268355  0.01777377 0.01827069
 0.01372255 0.01831213 0.01399318 0.01131341 0.01750056]

 GradientBoostingRegressor
 0
score:  -0.013730853769859941
mae:  2.3743744993605267
1
score:  -0.011192830275181143
mae:  0.8308633574876553
2
score:  -0.007819694775002661
mae:  2.401318291631377
3
score:  -0.013714149598559322
mae:  1.5314274301025577
[0.006931   0.02562932 0.02239726 0.02453456 0.03010415 0.01242182
 0.03193717 0.02015938 0.02534846 0.02579124 0.02527823 0.03852547
 0.02855066 0.0579799  0.01971772 0.02031744 0.02434095 0.04699475
 0.02001235 0.05813969 0.05101629 0.04349331 0.0221086  0.02564113
 0.01297282 0.017733   0.04591528 0.02354717 0.04030573 0.03027308
 0.00641122 0.012305   0.02894815 0.03015562 0.01046157 0.03360051
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.        ]

 RandomForestRegressor

 0
score:  -0.013730853769859941
mae:  2.3743744993605267
1
score:  -0.011192830275181143
mae:  0.8308633574876553
2
score:  -0.007819694775002661
mae:  2.401318291631377
3
score:  -0.013714149598559322
mae:  1.5314274301025577
(10000, 4)
[0.006931   0.02562932 0.02239726 0.02453456 0.03010415 0.01242182
 0.03193717 0.02015938 0.02534846 0.02579124 0.02527823 0.03852547
 0.02855066 0.0579799  0.01971772 0.02031744 0.02434095 0.04699475
 0.02001235 0.05813969 0.05101629 0.04349331 0.0221086  0.02564113
 0.01297282 0.017733   0.04591528 0.02354717 0.04030573 0.03027308
 0.00641122 0.012305   0.02894815 0.03015562 0.01046157 0.03360051
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.        ]

Decision Tree

 0
score:  -0.020894666828637387
mae:  2.373791104759805
1
score:  -0.007963529796098934
mae:  0.8313389053393785
2
score:  -0.020370236802909725
mae:  2.420878167676865
3
score:  -0.01426995728561642
mae:  1.5261941093237883
[0.         0.         0.         0.00249662 0.11334899 0.
 0.12136225 0.10666495 0.09691415 0.         0.         0.
 0.         0.00911951 0.         0.         0.         0.
 0.         0.         0.         0.02987637 0.06040505 0.03948063
 0.         0.         0.         0.         0.10946241 0.05104018
 0.         0.00030665 0.25952224 0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.        ]
'''