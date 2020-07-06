from keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# 1. 데이터 

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 20000)

# print(x_train.shape)  # (25000, )
# print(x_test.shape)   # (25000, )  
# print(y_train.shape)  # (25000, )
# print(y_test.shape)   # (25000, )
# print(y_train[0])     # (0)
# print(y_train[1])     # (1)

# 카테고리의 개수
category = np.max(y_train) + 1
# print(category)

# y의 유니크한 값 출력
y_b = np.unique(y_train)
# print(y_b)

y_train_pd = pd.DataFrame(y_train)
b = y_train_pd.groupby(0)[0].count()
# print(b.shape)   # (2, )

x_train = pad_sequences(x_train, padding = 'post')
x_test = pad_sequences(x_test, maxlen = 2494, padding = 'post')

# print(len(x_train[1]))   # 2494
# print(len(x_test[1]))      # 2315

# 2. 모델

from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Embedding, MaxPooling1D, Conv1D

model = Sequential()
model.add(Embedding(2000, 32, input_length = 2494))
model.add(Conv1D(128, 5, activation = 'relu'))
model.add(Flatten())
# model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# 3. 컴파일

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
history = model.fit(x_train, y_train, epochs = 75, batch_size = 320, validation_split = 0.2)

acc= model.evaluate(x_test, y_test, batch_size = 100)[1]
print("acc: ", acc)

# 4. 시각화

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker = '.', c = 'red', label = 'TestSet Loss')
plt.plot(y_loss, marker = '.', c = 'blue', label = 'TrainSet Loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()