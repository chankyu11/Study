## p.208

import numpy as np

arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

## 3행, 2행, 0행을 순서대로 추출하여 새로운 요소를 만듭니다

print(arr[[3, 2, 0]])

arr = np.arange(25).reshape(5, 5)
print(arr[[1, 3, 0]])

