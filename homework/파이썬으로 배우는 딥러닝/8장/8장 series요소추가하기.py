## p.235
## 요소추가

## series에 요소를 추가하려면 해당 요소도 series여야함.
## series형으로 변환하고 series에 append()로 전달해야 추가 실현

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index=index)

pineapple = pd.Series([12], index = ['pineapple'])
series = series.append(pineapple)

## 위와 동일
# series = series.append(pd.Series([12],index = ['pineapple']))
# series = series.append(pd.Series({'pineapple': 12}))

print(series)

