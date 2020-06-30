## p. 243
## 행추가

## df.append("Series형 데이터", ignore_index = True)를 실행하면 전달.
## 추가하고 데이터의 인덱스가 일치하지 않으면 NaN값으로 채워짐.

import pandas as pd

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
series = pd.Series(["mango", 2008, 7], index=["fruits", "year", "time"])

df = df.append(series, ignore_index=True)
print(df)


data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
series = pd.Series(["mango", 2008, 7], index=["fruits", "year", "time"])

df = df.append(series, ignore_index=True)
print(df)


index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
data3 = [30, 12, 10, 8, 25, 3]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)

# df에 series3을 추가해 df에 다시 대입
index.append("pineapple")
series3 = pd.Series(data3, index=index)
df = pd.DataFrame([series1, series2])

# df에 다시 대입
df = df.append(series3, ignore_index=True)
print(df)