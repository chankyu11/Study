import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i + size)]
        aaa.append([j for j in subset])
    return np.array(aaa)


size = 6

#1. 데이터

samsung = np.load('./test_samsung/npy/samsung.npy', allow_pickle='True')
hite = np.load('./test_samsung/npy/hite.npy', allow_pickle='True')

# print(samsung.shape) #509,1
# print(hite.shape)    #509,5
samsung = samsung.reshape(samsung.shape[0], ) # (509,)
samsung = (split_x(samsung,size))
# print(samsung.shape)   #(504,6)

scale = StandardScaler()
scale.fit(samsung)
scaled1 = scale.transform(samsung)
# print(scaled1)
# print(scaled1.shape) # (504, 6)

scale.fit(hite)
scaled2 = scale.transform(hite)
# print(scaled2)
# print(scaled2.shape) # (509, 5)

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 0]

# PCA
pca = PCA(n_components = 6)
pca.fit(hite)
x_hit = pca.transform(hite)
print(x_hit.shape) # (509, 1)
print(x_hit)
# print(x_sam.shape) #(504,5)
# print(y_sam.shape) #(504,)

# x_hit = x_hit[5:510, :, ]
# print(x_hit.shape)
# x_sam = x_sam.reshape(504,5,1)
# print(x_sam)
# x_hit = x_hit.reshape(x_hit.shape[0],x_hit.shape[1],1)
# print("x_hit: ", x_hit.shape)  # (504, 1, 1)

# #2.모델 구성

# input1 = Input(shape=(5,1))
# x1 = LSTM(100)(input1)
# x1 = Dense(100)(x1)
# x1 = Dense(100)(x1)
# x1 = Dense(100)(x1)
# x1 = Dense(100)(x1)

# input2 = Input(shape=(1,1))
# x2 = LSTM(100)(input2)
# x2 = Dense(100)(x2)
# x2 = Dense(100)(x2)
# x2 = Dense(100)(x2)
# x2 = Dense(100)(x2)

# merge = Concatenate()([x1, x2])

# output1 = Dense(10)(merge)
# output2 = Dense(1)(output1)

# model = Model(inputs = [input1, input2], outputs = output2)

# model.summary()

# #3.컴파일. 훈련
# model.compile(optimizer='adam', loss= 'mse', metrics=['mse'])
# model.fit([x_sam, x_hit], y_sam, epochs = 50)

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2, verbose=1,
#             batch_size=32, epochs = 1000000)

# loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size = 32)
# print('loss: ', loss)
# print('mse: ', mse)

# y_pred = model.predict([x1_test_scaled, x2_test_scaled])

# for i in range(5):
#     print('종가 : ', y1_test[i],'/ 예측가 :', y_pred[i])