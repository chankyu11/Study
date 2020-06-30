## p.239
## 정렬

## ???.sort_index()로 인덱스 정렬
## ???.sort_values()로 데이터 정렬
## 인수를 지정하지 않으면 기본값인 오름차순으로 정렬.
## ascending = False라고 지정하면 내림차순.

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

item1 = series.sort_index()
item2 = series.sort_values()

print(item1)
print(item2)
