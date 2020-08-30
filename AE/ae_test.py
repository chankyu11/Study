from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.models import Sequential

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters = hidden_layer_size, kenel_size = (3,3), padding = 'same',
                    input_shape = (28,28,1), 
                    activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(units = 784, activation = 'sigmoid'))
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()

x_train, y_train = train_set
x_test, y_test = test_set

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) / 255.
print(x_train.shape)
print(x_test.shape)

from keras.layers import MaxPooling2D