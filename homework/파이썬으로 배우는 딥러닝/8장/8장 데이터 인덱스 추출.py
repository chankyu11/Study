## p.233
## 데이터와 인덱스 추출

## series.values 로 데이터값을 참조 가능
## series.index 로 인덱스 참조 가능

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)
print(series)

series_values = series.values
print(series_values)

series_index = series.index
print(series_index)

