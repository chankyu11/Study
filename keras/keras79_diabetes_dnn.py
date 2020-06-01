
import numpy as np
from sklearn.datasets import load_diabetes
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Conv2D, Dropout
from keras.layers import MaxPooling2D, Flatten
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, StandardScaler
from sklearn.decomposition import PCA

diabetes = load_diabetes()

x = diabetes.data
y = diabetes.target

print(x.shape) # (442, 10)
print(y.shape) # (442, )
# print(x)
# print(y)

# scale

scale = StandardScaler()
scale.fit(x)
scaled = scale.transform(x)

# pca
pca = PCA(n_components = 5)
pca.fit(scaled)
x_pca = pca.transform(x)

# print(x.shape) # (442, 10)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, random_state = 123
                , train_size = 0.8)

print(x_train.shape) # (353, 5)
print(x_test.shape)  # (89, 5)
print(y_train.shape) # (353,)
print(y_test.shape)  # (89)

# 2. 모델

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape = (5, )))
model.add(Dense(150))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(90))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(1, activation = 'relu'))

model.summary()

# 3. 훈련, 컴파일

# es = EarlyStopping(monitor='val_loss', patience= 20, mode = 'auto')

# modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
# mcp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
#                         mode = 'auto', save_best_only = True)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs =100, batch_size= 64,
                validation_split = 0.2, verbose = 2)
                # callbacks = [es, mcp])


#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
print('loss', loss)
print('acc', acc)


'''
loss 2922.5540702905546
acc 2922.553955078125 
'''