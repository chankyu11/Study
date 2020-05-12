from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([10,20,30])
y_train = np.array([40,80,120])
# x_test = np.array([101,102,103,104,105,106,107,108,109,110])
# y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim = 1, activation = 'relu'))
model.add(Dense(3))
model.add(Dense(1, activation = 'relu'))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

loss, acc = model.evaluate(x_train, y_train, batch_size = 1)

print("loss: ", loss)
print("acc : ", acc)

output = model.predict(x_train)
print("결과물 : \n", output)
