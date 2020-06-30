## p.218
## 연습문제

import numpy as np
'''
arr = np.random.seed(100)

arr = np.random.randint(0,30,(5,3))
print(arr)

arr = np.random.randint(0,30,(5,3)).T
print(arr)
print(arr.shape)

arr1 = arr[:,2:5]
print(arr1.shape)
print(arr1)

# arr1 = np.sort(arr1)
arr1.sort()
print(arr1)

print(arr1.mean(axis = 0))
 '''
"======================================================================"

np.random.seed(0)

def make_image(m,n):
    image = np.random.randint(0,6,(m,n))
    return image

def change_little(matrix):
    shape = matrix.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.randint(0, 2) == 1:
                matrix[i][j] = np.random.randint(0, 6, 1)
    return matrix

image1 = make_image(3,3)
print(image1)

image2 = change_little(np.copy(image1))
print(image2)

image3 = image1 - image2
print(image3)

image3 = np.abs(image3)
print(image3)
