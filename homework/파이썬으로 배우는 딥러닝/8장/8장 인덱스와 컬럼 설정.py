## p.242
## 인덱스와 컬럼 설정

## df.index에 행 수와 같은 길이의 리스트를 대입하여 설정.
## df.colums에 열수와 같은 길이의 리스트를 대입하여 설정.

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)
df = pd.DataFrame([series1, series2])

# df의 인덱스가 1부터 시작하도록 설정
df.index = [1,2]
"========================================================================"
