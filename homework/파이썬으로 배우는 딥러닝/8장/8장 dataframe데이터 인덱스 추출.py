## p.233
## 데이터와 인덱스 추출

## series.values 로 데이터값을 참조 가능
## series.index 로 인덱스 참조 가능

import pandas as pd
import numpy as np

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)
print(series)

series_values = series.values
print(series_values)

series_index = series.index
print(series_index)


np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)

df.index = range(1,11)

df = df.loc[range(2,6), ["banana", "kiwifruit"]]

print(df)


## iloc

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)
print(df)
print("==="*10)

df = df.iloc[[1, 3], [0, 2]]
print(df)

np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

print(df)

df = df.iloc[range(1,5), [2, 4]]

print(df)