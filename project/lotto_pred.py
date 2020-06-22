import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.stats import itemfreq
from scipy.stats import mode

lotto = pd.read_csv('./project/2020-6-15lotto_data.csv', sep = ",", index_col = 0, header = None)

lo = lotto.iloc[:,0:6].values
x = lotto.iloc[:454,0:6].values
y = lotto.iloc[455:909,0:6].values
pred = lotto.iloc[910:, 0:6].values
# print(train)
# print(test)

x1 = x[:227, :]
y = x[227:, :]

# print(x1.shape)
# print(y.shape)
# print(x1)
# print("====" * 20)
# print(y)
x1 = x1.reshape(x1.shape[0], 6, 1)
# y = y.reshape(y.shape[0], y.shape[1], 1)

print(x1.shape)
# # print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x1, y, shuffle = False)

print(x_train.shape)
print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# 2. 모델

model = Sequential()
model.add(LSTM(128, activation = 'relu',return_sequences = True ,input_shape = (6, 1)))
# model.add(LSTM(64, activation = 'relu', input_shape = (6, 1)))
model.add(LSTM(25, activation = 'relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(6))

model.summary()

# 3. 실행

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 150, batch_size = 32, validation_split = 0.1)

# 4. 예측
pred = pred.reshape(5,6,1)
l1 = model.evaluate(x_test, y_test)
y_predict = model.predict(pred)
print(np.around(y_predict))
# print(y_predict)
print(l1)

lo1 = lo.reshape(5490, )
cnt = Counter(lo1)
print("최빈값: ", cnt.most_common(6))

# # 5. 시각화

# import matplotlib.pyplot as plt

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc =  hist.history['val_acc']

# # print('acc: ', acc)
# # print('val_acc: ', val_acc)

# plt.figure(figsize=(10 ,5))

# plt.subplot(2, 1, 1)
# plt.plot(hist.history['loss'], marker=',', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker=',', c='blue', label='val_loss')
# #plt.plot(hist.history['acc'])
# #plt.plot(hist.history['val_acc'])
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right') 
# # plt.show()

# plt.subplot(2, 1, 2)
# #plt.plot(hist.history['loss'])
# #plt.plot(hist.history['val_loss'])
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.grid() #  그림에  画格子
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc','val_acc'])

# plt.show()

# from matplotlib import style
# style.use('ggplot')

# figure = ['1','2','3', '4', '5',
#             '5', '6', '7', '8', '9',
#             '11', '12', '13', '14', '15',
#             '16', '17', '18', '19', '20',
#             '21', '22', '23', '24', '25',
#             '26', '27', '28', '29', '30',
#             '31', '32', '33', '34', '35',
#             '36', '37', '38', '39', '40',
#             '41', '42', '43', '44', '45']


# number = [128, 120, 119, 124, 127, 115, 121, 124, 93, 125, 
#                 124, 130, 128, 130, 122, 120, 132, 132, 128, 128,
#                 123, 102, 108, 121, 118, 124, 135, 114, 111, 111,
#                 123, 106, 127, 143, 112, 119, 127, 122, 131, 132,
#                 110, 118, 133, 120, 130]

# fig = plt.figure(figsize=(14, 9))
# ax = fig.add_subplot(111)

# ypos = np.arange(45)
# rects = plt.barh(ypos, number, align='center', height=0.4)
# plt.yticks(ypos, figure)

# plt.title('frequency')

# plt.tight_layout()
# plt.show()

# '''
# [[ 5.8391547 10.922296  18.736675  24.697065  31.79913   39.32152  ]
#  [ 5.6356225 10.414009  18.130962  23.91167   30.972054  38.416023 ]
#  [ 6.426335  10.830746  18.86303   24.812193  31.288876  38.98029  ]
#  [ 5.5452256  9.5561905 17.61874   23.135107  30.710123  38.192204 ]]

#  [[ 7. 14. 20. 26. 32. 38.]
#  [ 7. 13. 19. 25. 31. 37.]
#  [ 8. 13. 20. 26. 32. 38.]
#  [ 7. 12. 19. 25. 31. 37.]]

# epochs = 500
#  [[ 7. 24. 29. 31. 37. 42.]
#  [ 5. 10. 22. 27. 33. 43.]
#  [ 8. 12. 25. 32. 36. 33.]
#  [ 9. 14. 32. 37. 41. 44.]]
# '''