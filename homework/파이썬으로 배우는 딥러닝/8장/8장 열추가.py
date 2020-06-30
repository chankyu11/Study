
## p.245
## 열추가

## df["새로운 컬럼"] series 또는 리스트에 대입해서 추가.

import pandas as pd

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
series = pd.Series(["mango", 2008, 7], index=["fruits", "year", "time"])

df = df.append(series, ignore_index=True)
print(df)


index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)

new_column = pd.Series([15, 7], index=[0, 1])

# series1, seires2로 DataFrame을 생성합니다
df = pd.DataFrame([series1, series2])

# df에 새로운 열 "mango"를 만들어 new_column의 데이터를 추가하세요
df["mango"] = new_column
print(df)