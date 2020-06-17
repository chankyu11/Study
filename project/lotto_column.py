import numpy as np
import pandas as pd

lotto = pd.read_csv('./project/2020-6-15lotto_data.csv', index_col = None, header = None)

lotto.columns = ['회차', '1번', '2번', '3번', '4번', '5번', '6번', '보너스', '1등 당첨액', '2등 당첨액', '3등 당첨액', '4등 당첨액', '5등 당첨액']

lotto = lotto.sort_values(['회차'], ascending = [True])
lt = lotto.values

lt = lt[:, 1:7]

print(lt.shape) # (915, 6)

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i + size)]
        aaa.append([j for j in subset])
    return np.array(aaa)

size = 5

x = (split_x(lt,size))

print(x)
# print(x.shape)  # (911, 5, 6)

