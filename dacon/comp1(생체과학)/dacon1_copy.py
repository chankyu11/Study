import pandas as pd                         # 데이터 분석 패키지
import numpy as np                          # 계산 패키지
import matplotlib.pyplot as plt             # 데이터 시각화 패키지
import seaborn as sns                       # 데이터 시각화 패키지
import xgboost as xgb                       # XGBoost 패키지
from sklearn.model_selection import KFold   # K-Fold CV
import warnings
warnings.filterwarnings(action='ignore') 
import matplotlib
import sklearn


train = pd.read_csv('./dacon/comp1/train.csv')
test = pd.read_csv('./dacon/comp1/test.csv')
sm = pd.read_csv('./dacon/comp1/sample_submission.csv')

# print(train.shape)    # 10000, 76
# print(test.shape)     # 10000, 72

dst_columns = [k for k in train.columns if 'dst' in k]
train_dst = train[dst_columns]
test_dst = test[dst_columns]

train[dst_columns] = train_dst.interpolate(axis=1)
train.fillna(0, inplace=True)

test[dst_columns] = test_dst.interpolate(axis=1)
test.fillna(0, inplace=True)

x_train = train.loc[:, '650_dst':'990_dst']
y_train = train.loc[:, 'hhb':'na']
# print(x_train.shape, y_train.shape)   # 10000, 35, # 10000, 4


def train_model(x_data, y_data, k=5):
    models = []
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=123)
    
    for train_idx, val_idx in k_fold.split(x_data):
        x_train, y_train = x_data.iloc[train_idx], y_data[train_idx]
        x_val, y_val = x_data.iloc[val_idx], y_data[val_idx]
    
        d_train = xgb.DMatrix(data = x_train, label = y_train)
        d_val = xgb.DMatrix(data = x_val, label = y_val)
        
        wlist = [(d_train, 'train'), (d_val, 'eval')]
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'seed':777
            }

        model = xgb.train(params=params, dtrain=d_train, num_boost_round=500, verbose_eval=500, evals=wlist)
        models.append(model)
    
    return models

models = {}
for label in y_train.columns:
    print('train column : ', label)
    models[label] = train_model(x_train, y_train[label])
    print('\n\n\n')

for col in models:
    preds = []
    for model in models[col]:
        preds.append(model.predict(xgb.DMatrix(test.loc[:, '650_dst':])))
    pred = np.mean(preds, axis=0)

    sm[col] = pred

sm.to_csv('Dacon_baseline.csv', index=False)