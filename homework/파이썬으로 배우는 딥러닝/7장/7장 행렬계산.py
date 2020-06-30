## p.213
## 행렬 계산

## 두 행렬의 곱 반환 np.dot(a,b) 
## 노름(백터의 길이)을 반환하는 np.linalg.norm(a)

import numpy as np

arr = np.arange(9).reshape(3, 3)

# arr와 arr의 행렬곱
print(np.dot(arr, arr))

# vec로 변경
vec = arr.reshape(9)

# 변수 vec의 노름
print(np.linalg.norm(vec))

