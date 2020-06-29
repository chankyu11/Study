import numpy as np
import pandas as pd
from pandas import DataFrame

# NaN 값 제거
samsung = pd.read_csv("./data/CSV/samsung.csv", index_col = 0, header = 0, encoding='cp949' ,sep=',')
# print(samsung.shape)  # (700, 1)
samsung1 = samsung.dropna(how = 'all')
samsung2 = samsung1.drop(samsung1.index[0])  # (508, 1)
# print(samsung2)

hite = pd.read_csv("./data/CSV/hite.csv", index_col = 0, header = 0, encoding='cp949' ,sep=',')
hite1 = hite.dropna(how = 'all')
hite2 = hite1.fillna(0)
hite3 = hite2.drop(hite2.index[0])  # (508, 5)
# print(hite3)

def remove_comma(x):
    return x.replace(',','')

samsung2["시가"] = samsung2["시가"].apply(remove_comma)

hite3["시가"] = hite3["시가"].apply(remove_comma)
hite3["고가"] = hite3["고가"].apply(remove_comma)
hite3["저가"] = hite3["저가"].apply(remove_comma)
hite3["종가"] = hite3["종가"].apply(remove_comma)
hite3["거래량"] = hite3["거래량"].apply(remove_comma)

# print(hite3.dtypes)
# print(samsung2.dtypes)

samsung2 = samsung2.astype({"시가": np.int64})
# print(samsung2)
# print(samsung2.dtypes)
hite3 = hite3.astype({"시가" : np.int64,
                      "고가" : np.int64,  
                      "저가" : np.int64,
                      "종가" : np.int64,
                      "거래량" : np.int64})

# print(hite3)
# print(hite3.dtypes)
# for i in range(len(samsung.index)):
#     samsung.iloc[i,0] = int(samsung.iloc[i,0])

# for i in range(len(hite3.index)):
#     for j in range(len(hite3.iloc[i])):
#         hite.iloc[i,j] = int(hite3.iloc[i,j])
    
hite = hite3.sort_values(['일자'], ascending=[True])
samsung = samsung2.sort_values(['일자'], ascending=[True])

print(hite)
print(samsung)

# samsung = samsung.values
# hite = hite.values

# # 저장
# np.save('./data/samsung_data.npy', arr = samsung)
# np.save('./data/hite_data.npy', arr = hite)




# # 불러오기
# samsung_data = np.load('./data/samsung_data.npy')
# hite_data = np.load('./data/hite_data.npy')

# # print(samsung_data)
# # print(hite_data)
# # print(samsung_data.shape)  # (508, 1)
# # print(hite_data.shape)     # (508, 5) 

# import numpy as np
# from keras.models import Sequential, Model
# from keras.layers import Dense, LSTM, Input
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import robust_scale, StandardScaler
# from sklearn.decomposition import PCA


def split_xy5(ds, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(ds)):
        x_end_number = i + time_steps 
        y_end_number = x_end_number + y_column

        if y_end_number > len(ds):
            break
        tmp_x = ds[i:x_end_number, :]
        tmp_y = ds[x_end_number:y_end_number , 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung_data, 5, 1)
x2, y2 = split_xy5(hite_data, 5, 1)
# print(x2[0,:], "\n", y2[0])
# print(x1.shape) # 507 , 1, 1
# print(y1.shape) # 507 , 1, 1
# print(x2.shape) # (499, 5, 5)???
# print(y2.shape) # (499, 5, 5)???

# x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=1, test_size = 0.2)
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=1, test_size = 0.2)

# print(x1_train.shape)   # 402, 5, 5
# print(x1_test.shape)    # 101, 5, 5
# print(x2_train.shape)   # 405, 1, 1
# print(x2_test.shape)    # 102, 1, 1
# print(y1_train.shape)   # 402, 1, 1
# print(y1_test.shape)    # 101, 1, 1
# print(y2_train.shape)   # 405, 1, 1
# print(y2_test.shape)    # 102, 1 ,1

# x1_train = np.reshape(x1_train,(x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
# x1_test = np.reshape(x1_test,(x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
# x2_train = np.reshape(x2_train,(x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
# x2_test = np.reshape(x2_test,(x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))

# print(x1_train.shape) # (402, 25)
# print(x1_test.shape)  # (101, 25)
# print(x2_train.shape) # (405, 1)
# print(x2_test.shape)  # (102, 1)

# # scaler

# scaler = StandardScaler()
# scaler.fit(x1_train)
# x1_train_scaled = scaler.transform(x1_train)
# x1_test_scaled = scaler.transform(x1_test)

# scaler2 = StandardScaler()
# scaler2.fit(x2_train)
# x2_train_scaled = scaler2.transform(x2_train)
# x2_test_scaled = scaler2.transform(x2_test)

# # x1_train_scaled = x1_train_scaled.reshape(402,5,5)
# # x1_test_scaled = x1_test_scaled.reshape(101,5,5)
# # x2_train_scaled = x2_train_scaled.reshape(405,1,1)
# # x2_test_scaled = x2_test_scaled.reshape(102,1,1)


# print(x1_train_scaled.shape)
# input1 = Input(shape = (25,))
# dense1 = LSTM(128)(input1)
# dense2 = Dense(64)(dense1)
# dense3 = Dense(32)(dense2)
# dense4 = Dense(32)(dense3)
# dense5 = Dense(16)(dense4)

# input2 = Input(shape = (1, ))
# dense1_1 = LSTM(128)(input2)
# dense2_1 = Dense(64)(dense1_1)
# dense3_1 = Dense(32)(dense2_1)
# dense4_1 = Dense(32)(dense3_1)
# dense5_1 = Dense(16)(dense4_1)

# from keras.layers import concatenate

# merge = concatenate([dense5, dense5_1])
# # output = Dense(100)(merge)
# output1 = Dense(1)(merge)

# output2 = Dense(1)(merge)


# model = Model(inputs = [input1, input2], outputs = [output1, output2])

# model.summary()

# # early_stopping = EarlyStopping(patience=20)
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit([x1_train_scaled, x2_train_scaled], [y1_train, y2_train], validation_split=0.2, verbose=1,
#             batch_size=32, epochs=100)

# loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], [y1_test, y2_test], batch_size=1)
# print('loss: ', loss)
# print('mse: ', mse)

# y_pred = model.predict([x1_test_scaled, x2_test_scaled])

# for i in range(5):
#     print('종가 : ', y1_test[i],'/ 예측가 :', y_pred[i])

