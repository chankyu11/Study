## p.237
## 필터링

## Series형 데이터에서 조건과 일치하는 요소를 꺼내고 싶을때 사용.
## 
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

conditions = [True, True, False, False, False]

print(series)
print(series[conditions])

## 당연히 >=5 라는 조건도 당연히 가능.
print(series[series >= 5])

## [][] 这样可使复数
print(series[series < 12][series >4])

