import tensorflow as tf
import numpy as np

ds = np.array([1,2,3,4,5,6,7,8,9,10])
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(ds, size)

print(dataset.shape) # 6,5
print(dataset)