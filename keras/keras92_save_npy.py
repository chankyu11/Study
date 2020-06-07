from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

print(type(iris))

x_data = iris.data
y_data = iris.target

print(type(x_data))

np.save('./data/iris_x.npy', arr = x_data)
np.save('./data/iris_y.npy', arr = y_data)

x_data_load = np.load('./dataD/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')

print(type(x_data_load))
print(type(y_data_load))
print(x_data_load.shape) # (150, 4)
print(y_data_load.shape) # (150, )

print(x_data)
print(y_data)