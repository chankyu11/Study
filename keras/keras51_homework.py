# 49번에 대한 답
import numpy as np

# 1. 
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1
print(y) 

# 2. 
x = np.array([1,2,3,4,5,1,2,3,4,5])
print(x.shape)
x = x.reshape(10, 1)
#  =  y = y.reshape(-1, 1)

from sklearn.preprocessing import OneHotEncoder
x1 = OneHotEncoder()
x1.fit(x)
x1 = x1.transform(x).toarray()
print(x1)
# 3.
# y = np.array([1,2,3,4,5,1,2,3,4,5])

# from keras.utils import np_utils
# y = np_utils.to_categorical(y)
# print(y)
# print(y.shape)

# 4.
# y = y[:,1:]
# print(y)
# print(y.shape)
