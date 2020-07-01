## p.252
## 행 또는 열 삭!제!

## df.drop()으로 삭제 가능

import pandas as pd
import numpy as np

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
        
df = pd.DataFrame(data)
print(df)

df_1 = df.drop(range(0,2))
print(df_1)

df_2 = df.drop("year", axis = 1)
print(df_2)


np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
df = pd.DataFrame()

for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# drop()을 이용하여 df에서 홀수 인덱스가 붙은 행만을 남겨 df에 대입하세요
df = df.drop(np.arange(2,11,2))
## np.arange(2, 11, 2)는 2에서 10까지의 수열을 2의 간격으로 추출한 것
## 출력은 2,4,6,8,10
## np.arange(2, 11 ,3)는 2에서 10까지 3의 간력으로 추출.

df = df.drop("strawberry", axis = 1)
print(df)