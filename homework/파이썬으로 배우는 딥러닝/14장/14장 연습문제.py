import pandas as pd
import numpy as np
from numpy import nan as NA
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

# 컬럼 추가합니다
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
            "Magnesium", "Total phenols", "Flavanoids",
            "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
            "OD280/OD315 of diluted wines","Proline"]

# df의 상위 10행 변수 df_ten에 대입, 표시
df_ten = df.head(10)
print(df_ten)

# 데이터 누락
df_ten.iloc[1,0] = NA
df_ten.iloc[2,3] = NA
df_ten.iloc[4,8] = NA
df_ten.iloc[7,3] = NA
print(df_ten)

# 평균값 대입
df_ten.fillna(df_ten.mean())
print(df_ten)

# "Alcohol" 열의 평균
print(df_ten["Alcohol"].mean())

# 중복된 행 제거
df_ten.append(df_ten.loc[3])
df_ten.append(df_ten.loc[6])
df_ten.append(df_ten.loc[9])
df_ten = df_ten.drop_duplicates()
print(df_ten)

# Alcohol 열의 구간 리스트 작성
alcohol_bins = [0,5,10,15,20,25]
alcoholr_cut_data = pd.cut(df_ten["Alcohol"],alcohol_bins)

# 구간 수를 집계, 출력
print(pd.value_counts(alcoholr_cut_data))
