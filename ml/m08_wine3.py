import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기

wine = pd.read_csv('./data/CSV/winequality-white.csv', sep = ';', header = 0)

wine.groupby('quality')['quality'].count()
# 열 안에 있는 객체들. 행 별로 숫자를 数 
# groupby

count_data = wine.groupby('quality')['quality'].count()

print(count_data)

count_data.plot()
plt.show()