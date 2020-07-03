import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, StandardScaler
from sklearn.decomposition import PCA

breast_cancer = load_breast_cancer()

x = breast_cancer.data
y = breast_cancer.target

print(x.shape) # (569, 30)
print(y.shape) # (569, )
# print(x)
# print(y)

scaler = StandardScaler()
scaler.fit(x)
x_sca = scaler.transform(x)

pca = PCA()
x_pca = pca.fit_transform(x_sca)
print(x_pca.shape) # (569, 30)

x = x_pca.reshape(x_pca.shape[0], 30 ,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 30, train_size = 0.8)

print(x_train.shape) #(455, 30, 1)
print(x_test.shape)  #(114, 30, 1)
print(y_train.shape) #(455, )
print(y_test.shape)  #(114, )

# 2. 모델

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (30,1)))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(5))
model.add(Dense(1, activation = 'sigmoid'))

# 3. 훈련, 컴파일

es = EarlyStopping(monitor='val_loss', patience= 20, mode = 'auto')

modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
                        mode = 'auto', save_best_only = True)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs =100, batch_size= 64, validation_split = 0.2, verbose = 2)
                # callbacks = [es, mcp])


#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
print('loss', loss)
print('acc', acc)

'''
loss 0.46736251209911545
acc 0.8245614171028137

'''