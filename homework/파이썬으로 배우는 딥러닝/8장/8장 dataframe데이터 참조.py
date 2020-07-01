## p.247
## 데이터 참조

## loc = 이름으로 참조
## iloc = 번호로 참조 (위치) 

import pandas as pd

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "time": [1, 4, 5, 6, 3],
        "year": [2001, 2002, 2001, 2008, 2006]}
        
df = pd.DataFrame(data)

print(df)

df = df.loc[[1,2], ["time", "year"]]
print(df)