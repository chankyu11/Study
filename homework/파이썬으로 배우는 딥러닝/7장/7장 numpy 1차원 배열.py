## p.187
## import 
 
import numpy as np

## 1차원 배열

np.array([1,2,3])
# print(np.arange(5))

a = np.array([[[1,2],[3,4],[5,6],[7,8]]])
# print(a.shape)  # (1,4,2)

## p.188

storages = [24, 3, 4, 23, 10, 12]
np_storages = np.array(storages)

print(type(np_storages))

## p.189
# 未使 numpy
storages = [1, 2, 3, 4]
new_storages = []
for n in storages:
    print(n)
    n += n
    print(n)
    new_storages.append(n)
# print(new_storages)

# 使numpy

storages = np.array([1, 2, 3, 4])
storages += storages
# print(storages)

## 문제

arr = np.array([2,5,3,4,8])

print(arr + arr)
print(arr - arr)
print(arr ** 3)
print(1 / arr)
