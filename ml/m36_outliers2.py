import numpy as np
import pandas as pd

def multi_check(data):
    multi_check = []
    for i in range(data.shape[1]):
        data1 = data[:,i]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        print("1사분위: ", quartile_1)
        print("3사분위: ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        a = np.where((data > upper_bound) | (data < lower_bound))
        multi_check.append(a)
    return multi_check             
a = np.array([1,2,3,4,10000,6,7,5000,90,100])
b = multi_check(a)
# print(a.shape)
# print(b)

train = np.load('D:/STUDY/dacon/comp1/train_bfill.npy')
# print(len(train))
a = multi_check(train)
print(a)