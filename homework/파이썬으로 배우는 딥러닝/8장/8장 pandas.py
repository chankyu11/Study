## p.225
## pandas
## series와 dataframe의 데이터 확인

## series는 딕셔너리형을 전달하면 {key: 값} 키에 의해 정렬됨.

import pandas as pd

# Series 데이터
fruits = {"banana": 3, "orange": 2}
print(pd.Series(fruits))

# DataFrame 데이터
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df)


## p.228

# Series용 라벨(인덱스)을 작성
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# 데이터를 대입
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index=index)

# 딕셔너리 형을 사용하여 DataFrame용 데이터를 작성
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)

print("Series 데이터")
print(series)
print("\n")
print("DataFrame 데이터")
print(df)
