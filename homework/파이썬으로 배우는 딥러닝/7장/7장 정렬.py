## p.211
## 정렬

import numpy as np

arr = np.array([15,30,5])
print(arr.argsort())

arr = np.array([[8, 4, 2], [3, 5, 1]])

## argsort()
print(arr.argsort())

## np.sort()
print(np.sort(arr))

## sort()
arr.sort(1)
print(arr)