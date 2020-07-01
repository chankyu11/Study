## p.237
## 요소삭제

## series.drop("인덱스")를 사용하여 삭제 가능.

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)
print(series)
series = series.drop("strawberry")
print(series)

