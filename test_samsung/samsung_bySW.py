import numpy as np
import pandas as pd
from pandas import DataFrame

# NaN 값 제거
samsung = pd.read_csv("./test_samsung/samsung.csv", index_col = 0, header = 0, encoding='cp949' ,sep=',')

hite = pd.read_csv("./test_samsung/hite.csv", index_col = 0, header = 0, encoding='cp949' ,sep=',')


samsung = samsung.dropna(axis= 0)
# axis = 0는 행
print(samsung.shape)

# 제거 방법1
hite = hite.fillna(method = 'bfill')
# ffill은 NaN을 위에 있는 수치로 채운다.
# bfill은 밑에 있는 수치로 nan을 채운다.
hite = hite.dropna(axis= 0)
# print(hite)


# 제거 방법2
hite = hite[0:509]
# hite.iloc[0, 1:5] = ['10', '20', '30', '40']
# #  [행, 열] 그리고 넣을 수치 iloc
hite.loc['2020-06-02', '고가':'거래량'] = ['100', '200', '300', '400']
# #  해더와 이름을 사용하는게 loc

# print(hite)

# 삼성과 하이트의 정렬을 오름차순으로 변경

# hite = hite.sort_values(['일자'], ascending=[True])
# samsung = samsung.sort_values(['일자'], ascending=[True])

# print(hite)
# print(samsung)
# 콤마 제거

for i in range(len(samsung.index)):
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))
print(samsung)
# print(type(samsung.iloc[0,0]))

for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i, j] = int(hite.iloc[i, j].replace(',',''))

hite = hite.sort_values(['일자'], ascending=[True])
samsung = samsung.sort_values(['일자'], ascending=[True])

print(hite)
print(samsung)

samsung = samsung.values
hite = hite.values

def split_xy5(ds, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(ds)):
        x_end_number = i + time_steps 
        y_end_number = x_end_number + y_column

        if y_end_number > len(ds):
            break
        tmp_x = ds[i:x_end_number, :]
        tmp_y = ds[x_end_number:y_end_number , 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung, 5, 1)

print(x1.shape)
print(x1[:5,:])
print(y1[:5,:])