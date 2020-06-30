## p. 199

## np.unique() = 중복 제거 정렬
## np.union1d(x, y) = x,y의 합집합 정렬
## np.untersect1d(x, y) = x,y 교집합 정렬
## np.setdiff1d(x, y) = x,y 차집합 정렬 

import numpy as np

arr1 = [2, 5, 7, 9, 5, 2]
arr2 = [2, 5, 8, 3, 1]

## 중복제거 정렬
new_arr1 = np.unique(arr1)
print(new_arr1)

## 합집합 
print(np.union1d(new_arr1, arr2))

## 교집합
print(np.intersect1d(new_arr1, arr2))

## 차집합
print(np.setdiff1d(new_arr1, arr2))




