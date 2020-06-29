import numpy as np
import time
from numpy.random import rand

N = 150

mata = np.array(rand(N,N))
matb = np.array(rand(N,N))
matc = np.array([0] * N for _ in range(N))

start = time.time()


for i in range(N):
    for j in range(N):
        for k in range(N):
            matc[i][j] = mata[i][k] * matb[k][j]
            

print("파이썬의 기능만으로 계산한 결과：%.2f[sec]" % float(time.time() -start))

start = time.time()

matC = np.dot(mata, matb)

print("NumPy를 사용한 경우의 계산 결과：%.2f[sec]" % float(time.time() - start))

'''
파이썬의 기능만으로 계산한 결과：4.43[sec]
NumPy를 사용한 경우의 계산 결과：0.00[sec]
'''