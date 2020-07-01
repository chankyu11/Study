## p.259
## 연습문제

import pandas as pd
import numpy as np

index = ["growth", "mission", "banana", "kiwifruit"]
data = [50, 7, 26, 1]

# Series를 작성하세요
series = pd.Series(data, index = index)

# 인덱스에 알파벳 순으로 정렬한 series를 aidemy에 대입하세요
aidemy = series.sort_index()
print(aidemy)

# 인덱스가 "tutor"이고 데이터가 30인 요소를 series에 추가하세요
a = pd.Series([30], index = ["tutor"])
a1 = series.append(a)

print(aidemy)
print(a)
print(a1)

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for index in index:
    df[index] = np.random.choice(range(1,11), 10)

df.index = range(1,11)

# loc[]를 사용하여 df의 2~5행(4개의 행)과 "banana","kiwifruit"를 포함하는 
# DataFrame을 aidemy3에 대입하세요
# 첫번째 행의 인덱스는 1이며, 이후의 인덱스는 정수의 오름차순입니다

aidemy3 = df.loc[range(2,6),["banana","kiwifruit"]]
print(aidemy3)