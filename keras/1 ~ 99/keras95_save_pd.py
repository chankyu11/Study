# CSV 파일을 보면 해더 부분 첫 행, 첫 열을 붙여줄지 뺄지 생각해야함.
# 150,4,setosa,versicolor,virginica iris 파일의 해더 부분.

import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/CSV/iris.csv", index_col = None, header = 0, sep=',')

print(datasets)
print(datasets.head()) # 위에서 부터 5개 정도만 보여줌
print(datasets.tail()) # 아래에서 5개 정도

print("======================" * 5)

aaa = datasets.values
print(type(aaa))

np.save('./data/iris_data.npy', arr = aaa)

iris_data_load = np.load('./data/iris_data.npy')

print(iris_data_load)
print(iris_data_load.shape)