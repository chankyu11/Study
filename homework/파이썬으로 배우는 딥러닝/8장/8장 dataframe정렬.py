## p.255
## pandas 정렬

## df형 변수에서 df.sort_values(by = "칼럼 또는 칼럼 리스트", ascending = True)
## 이렇게 지정하면 오름차순으로 정렬.
## 반대로 ascending  = False로 정하면 내림차순.

import pandas as pd
import numpy as np

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "time": [1, 4, 5, 6, 3],
        "year": [2001, 2002, 2001, 2008, 2006]}
df = pd.DataFrame(data)
print(df)

## 오름차순
df = df.sort_values(by="year", ascending = True)
print(df)

## 데이터를 오름차순으로 정렬(타임, 이얼)
df = df.sort_values(by = ["time", "year"], ascending = True)
print(df)

np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()

for column in columns:
    df[column] = np.random.choice(range(1,11),10)

df.index = range(1,11)

# df를 "apple", "orange", "banana", "strawberry" "kiwifruit"의 순으로 오름차순 정렬하세요
# 정렬한 결과로 만들어진 DataFrame을 df에 대입하세요. 첫번째 인수이면 by는 생략 가능합니다
df = df.sort_values(by=columns)

print(df)
