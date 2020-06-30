## p.240
## dataframe

## df는 여러 개 묶은 것 같은 2차원 데이터 구조를 하고 있음.
## pd.dataframe()에 series를 전달하여 생성.
## 딕셔너리, 리스트 형식으로 가능. 하지만 해당 리스트형의 길이는 필 동일.

import pandas as pd

data = {"fruits": ["apple", "orange", "banana", "strawberry",
"kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}

data = pd.DataFrame(data)

print(data)
print(type(data))

"======================================================================"

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)
print(series1)
print(series2)

df = pd.DataFrame([series1, series2])
print(df)
