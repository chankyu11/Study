# p. 417

import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

# 각 수치가 무엇을 나타내는지 컬럼 헤더로 추가합니다
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols",
"Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

# print(df)

'''
  Alcohol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  OD280/OD315 of diluted wines  Proline
0    1    14.23        1.71  2.43               15.6        127           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92     1065
1    1    13.20        1.78  2.14               11.2        100           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40     1050
2    1    13.16        2.36  2.67               18.6        101           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17     1185
3    1    14.37        1.95  2.50               16.8        113           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45     1480
4    1    13.24        2.59  2.87               21.0        118           2.80        2.69                  0.39             1.82             4.32  1.04                          2.93      735
..  ..      ...         ...   ...                ...        ...            ...         ...                   ...              ...              ...   ...                           ...      ...
173  3    13.71        5.65  2.45               20.5         95           1.68        0.61                  0.52             1.06             7.70  0.64                          1.74      740
174  3    13.40        3.91  2.48               23.0        102           1.80        0.75                  0.43             1.41             7.30  0.70                          1.56      750
175  3    13.27        4.28  2.26               20.0        120           1.59        0.69                  0.43             1.35            10.20  0.59                          1.56      835
176  3    13.17        2.59  2.37               20.0        120           1.65        0.68                  0.53             1.46             9.30  0.60                          1.62      840
177  3    14.13        4.10  2.74               24.5         96           2.05        0.76                  0.56             1.35             9.20  0.61                          1.60      560
'''

# p.418
# 아이리스 데이터 불러오기

df = pd.read_csv(
"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None)
df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

print(df)

'''
            5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa
..            ...          ...           ...          ...             ...
145           6.7          3.0           5.2          2.3  Iris-virginica
146           6.3          2.5           5.0          1.9  Iris-virginica
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica
'''
