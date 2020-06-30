## p.202

## np.array([리스트], [리스트])
## shape = 구조 확인
## reshape = 구조 변경


import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr)

print(arr.shape)

## 구조변경 (4,2)
print(arr.reshape(4, 2))


## 슬라이스

arr = np.array([[1, 2 ,3], [4, 5, 6]])
print(arr[1])

arr = np.array([[1, 2 ,3], [4, 5, 6]])
print(arr[1,2])

arr = np.array([[1, 2 ,3], [4, 5, 6]])
print(arr[1,1:])

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr) 

# 요소 중 3출력
print(arr[0, 2])

# 1행 ,2열 출력
print(arr[1:, :2])

## p.206
## axis

## axis = 좌표축!!!
## 열마다 처리는 = 0
## 행마다 처리는 = 1

## numpy에서는 .ndarray.sum()으로 모든 요소 더하기 가능


arr = np.array([[1, 2 ,3], [4, 5, 6]])

print(arr.sum())
print(arr.sum(axis=0))
print(arr.sum(axis=1))

arr = np.array([[1, 2, 3], [4, 5, 12], [15, 20, 22]])

print(arr.sum(axis=1))
print(arr.sum(axis=0))

