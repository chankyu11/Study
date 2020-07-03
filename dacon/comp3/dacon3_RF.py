import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, LSTM, MaxPooling1D, Conv1D, Flatten
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

test_features = pd.read_csv('./dacon/comp3/test_features.csv', header = 0, index_col = 0, encoding = 'cp949')
train_features = pd.read_csv('./dacon/comp3/train_features.csv', header = 0, index_col = 0, encoding = 'cp949')
train_target = pd.read_csv('./dacon/comp3/train_target.csv', header = 0, index_col = 0, encoding = 'cp949')
sample = pd.read_csv('./dacon/comp3/sample_submission.csv', header = 0, index_col = 0, encoding = 'cp949')

def preprocessing_KAERI(data) :
    '''
    data: train_features.csv or test_features.csv
    
    return: Random Forest 모델 입력용 데이터
    '''
    
    # 충돌체 별로 0.000116 초 까지의 가속도 데이터만 활용해보기 
    _data = data.groupby('id').head(30)
    
    # string 형태로 변환
    _data['Time'] = _data['Time'].astype('str')
    
    # Random Forest 모델에 입력 할 수 있는 1차원 형태로 가속도 데이터 변환
    _data = _data.pivot_table(index = 'id', columns = 'Time', values = ['S1', 'S2', 'S3', 'S4'])
    
    # column 명 변환
    _data.columns = ['_'.join(col) for col in _data.columns.values]
    
    return _data

train_features = preprocessing_KAERI(train_features)
test_features = preprocessing_KAERI(test_features)

import sklearn
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1, random_state=0)

# 모델 학습 (fit)
model.fit(train_features, train_target)

y_pred = model.predict(test_features)

submit = pd.read_csv('./dacon/comp3/sample_submission.csv')

for i in range(4):
    submit.iloc[:,i+1] = y_pred[:,i]