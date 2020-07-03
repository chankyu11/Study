import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM
from keras.callbacks import EarlyStopping

# 1. 데이터

a = np.array(range(1,101))
size = 5    # time steps = 4

# lstm 모델 완성.

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("=========================================")
print(dataset)

x = dataset[:,0:4]
y = dataset[:,4]

x = x.reshape(x.shape[0],x.shape[1],1)

# 2. 모델

model = Sequential()
model.add(LSTM(5, input_shape = 4, input_dim = 1))
model.add(Dense(3))
model.add(Dense(1))

# model.summary()

# 3. 훈련
from keras.callbacks import TensorBoard
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images= True)
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='loss', patience= 10, mode = 'auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x,y,epochs = 100,batch_size=1, verbose=1, callbacks = [es, tb_hist])

print(hist)
print(hist.history.keys())

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
# plt.plot(hist.add_history['val_loss'])
plt.title('loss & acc')
plt.ylabel('loss,acc')
plt.xlabel('epoch')
# plt.show()

'''
# 4. 평가, 예측
loss, mse = model.evaluate(x, y)

y_predict = model.predict(x)
print(y_predict)

print('loss', loss)
print('mse', mse)
'''
