## p.231
## 참조

import pandas as pd
fruits = {"banana": 3, "orange": 4, "grape": 1, "peach": 5}
series = pd.Series(fruits)
print(series[0:2])

print(series[["orange", "peach"]])

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

# 인덱스 참조로 series의 2~4번째 요소를 꺼내 items1에 대입
items1 = series[1:4]

# "apple", "banana", "kiwifruit"을 요소를 items2에 대입
items2 = series[["apple", "banana", "kiwifruit"]]
print(items1)
print()
print(items2)
