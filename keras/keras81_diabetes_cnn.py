
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
pca = PCA(n_components = 4)
pca.fit(scaled)
x_pca = pca.transform(x)

x_pca = x_pca.reshape(x_pca.shape[0], 2, 2, 1)
# print(x_pca.shape) # (442, 5, 1)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, random_state = 123
                , train_size = 0.8)

print(x_train.shape) # (353, 2, 2, 1)
print(x_test.shape)  # (89, 2, 2, 1)
print(y_train.shape) # (353,)
print(y_test.shape)  # (89)

# 2. 모델

model = Sequential()
model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = (2, 2, 1)))
model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(filters = 30, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 30, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))


# 3. 훈련, 컴파일

es = EarlyStopping(monitor='val_loss', patience= 20, mode = 'auto')

# modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
# mcp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
#                         mode = 'auto', save_best_only = True)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs =300, batch_size= 64,
                validation_split = 0.2, verbose = 2)
                # ,callbacks = [es])


#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
print('loss', loss)
print('acc', acc)

'''
loss 2697.5001947638693
acc 2697.500244140625

'''