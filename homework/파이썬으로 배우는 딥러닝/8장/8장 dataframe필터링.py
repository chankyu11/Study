## p.257
## dataframe 필터링

## dataframe도 series와 마찬가지 bool형의 시퀀스를 지정하여 True인 것만 추출 가능

import pandas as pd
import numpy as np

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)

print(df.index % 2 == 0) # index가 2로 나눠서 0인거
print()
print(df[df.index % 2 == 0]) # 0 ,2 ,4 번째 행 추출



np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11), 10)

df.index = range(1,11)


## 필터링을 사용하여 df의 apple 열에서 5이상, 키위열에서 5이상 값 df대입
df = df.loc[df["apple"] >= 5]
print(df)
df = df.loc[df["kiwifruit"] >= 5]
print(df)
#df = df.loc[df["apple"] >= 5][df["kiwifruit"] >= 5] 라도 OK


