## p.214
## 통계함수

## mean(), np.average() = 평균
## max() = 최댓값, min() = 최솟값
## np.argmax() = 요소의 최댓값, np.argmin() = 요소의 최솟값
## np.std() = 표준편차, np.var() = 분산
## 모두 aixs = ? 로 축 지정이 가능.

import numpy as np

arr = np.arange(15).reshape(3, 5)

print(arr)
## arr의 각 열의 평균
print(arr.mean(axis = 0))

## 행의 합
print(arr.sum(1))

## 변수의 최솟값 
print(arr.min())

## 변수 각 열의 최대값의 인덱스 번호
print(arr.argmax(axis = 0))

