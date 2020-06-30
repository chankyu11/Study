## p.200

##  randint()만 적어도 함수 사용가능하게
## np.random.rand()함수는 난수 생성함수
## random.randint(x,y,z)  x이상 y미만 z개 생성
## 가우스 분포를 따르는 난수생성 = np.random.normal()
## random.randint(x,y,z) z자리에 (2,3) 이렇게 대입해 (2,3)행렬도 생성가능

from numpy.random import randint
import numpy as np
## 변수1에 각 요소가 0이상 10 이하인 정수 행렬을 (5, 2) 대입
arr1 = randint(0, 11, (5, 2))
print(arr1)

## 변수2에 0이상 1미만의 난수 3개 생성
arr2 = np.random.rand(3)
print(arr2)

