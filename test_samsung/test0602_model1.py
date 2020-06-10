import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Input, LSTM, Dropout
from keras.layers import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)
size = 6

# 1. 데이터
# npy 불러오기

samsung = np.load('./test_samsung/npy/samsung.npy', allow_pickle=True)
hite = np.load('./test_samsung/npy/hite.npy', allow_pickle=True)

# print(samsung.shape)
# print(hite.shape)
samsung = samsung.reshape(samsung.shape[0],) 
print(samsung.shape) # (509, )

samsung = (split_x(samsung, size))
print(samsung.shape)  # (504, 6)
# 삼성을 6새씩 다르겠다는 의미.

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]
# 6일치가 y이다.

# print(x_sam.shape)  # (504, 5)
# print(y_sam.shape)  # (504, )

x_hit = hite[5:510, :] # (504, 5)
print(x_hit.shape)
# 2. 모델

input1 = Input(shape = (5,))
dense = Dense(10)(input1)
dense1 = Dense(10)(dense)

input2 = Input(shape= (5, ))
dense1 = Dense(15)(input2)
dense2 = Dense(10)(dense1)

merge = concatenate([dense1, dense2])

output = Dense(1)(merge)

model = Model(input = [input1, input2], output = output)

model.summary()

# 3. 컴파일, 훈련

model.compile(optimizer = 'adam', loss = 'mse')
model.fit([x_sam,x_hit], y_sam, epochs = 100)