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

train_featuers = np.load('./dacon/comp3/train_f1.npy')
test_featuers = np.load('./dacon/comp3/test_f1.npy')
train_target = np.load('./dacon/comp3/train_t1.npy')
sample = np.load('./dacon/comp3/sample1.npy')

# print(x)
# print(train_featuers.shape)           # (1050000, 4)  x
# print(train_target.shape)             # (2800, 4)  y
# print(test_featuers.shape)            # (262500,  4)  x_pred
# print(sample.shape)                   # (700,  4)  

x = train_featuers.reshape(2800, 375, 4)   
x_pred = test_featuers.reshape(700, 375, 4)
y = train_target
print(x_pred.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
# print(x_train.shape)
# print(x_test.shape)

# 2. 모델

# input1 = Input(shape = (375,4))
# dense2 = Conv1D(32, 2, padding = 'same',activation= 'relu')(input1)
# dense2 = Conv1D(64, 2, padding = 'same',activation= 'relu')(dense2)
# dense2 = MaxPooling1D()(dense2)
# dense2 = Conv1D(128, 2, padding = 'same',activation= 'relu')(dense2)
# dense2 = Conv1D(64, 2, padding = 'same',activation= 'relu')(dense2)
# dense2 = MaxPooling1D()(dense2)
# dense2 = Conv1D(32, 2, padding = 'same',activation= 'relu')(dense2)
# dense2 = Conv1D(16, 2, padding = 'same',activation= 'relu')(dense2)
# # dense1 = LSTM(10, activation= 'relu',return_sequences = True)(dense2)
# dense2 = MaxPooling1D()(dense2)
# dense2 = Flatten()(dense2)

# dense3 = Dense(128,activation='relu')(dense2)

# output1 = Dense(64)(dense3)
# output2 = Dense(16)(output1)
# output3 = Dense(4)(output2)

# model = Model(inputs = input1, outputs = output3)

model = Sequential()
model.add(Conv1D(32, 2, input_shape = (375,4), activation = 'relu'))
model.add(Conv1D(64, 2, input_shape = (375,4), activation = 'relu'))
model.add(Conv1D(128, 2, input_shape = (375,4), activation = 'relu'))
model.add(Conv1D(256, 2, input_shape = (375,4), activation = 'relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(4))

'''
result:  1524632804.5714285
y_pres:  [[ 3.9142373e+03 -1.5184637e+03  4.6696997e+02 -2.1017722e+03]
'''

# 3. 컴파일

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 200, validation_split = 0.25)

# 4. 평가

result = model.evaluate(x_test, y_test)
y_pred = model.predict(x_pred)

print("result: ", result)
print("y_pres: \n ", y_pred)

submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./dacon/comp3/sample_submission1.csv', index = False)