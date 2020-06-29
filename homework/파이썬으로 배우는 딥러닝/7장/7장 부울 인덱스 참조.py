import numpy as np
## p.196

arr = np.array([2, 4, 6, 7])
# print(arr[np.array([True, True, True, False])])

arr = np.array([2, 4, 6, 7])
print(arr[arr % 3 == 1])

arr = np.array([2, 3, 4, 5, 6, 7])
print(arr % 2 == 0)
print(arr[arr % 2 == 0])
