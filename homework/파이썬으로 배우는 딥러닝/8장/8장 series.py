## p.229
## series

import pandas as pd

fruits = {"banana": 3, "orange": 2}
print(pd.Series(fruits))

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)
print(series)
