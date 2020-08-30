<<<<<<< HEAD
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

ds = load_diabetes()

X = ds.data
Y = ds.target

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.
print(cumsum)

print(np.argmax(cumsum >= 0.94) + 1)
# print(cumsum >= 0.94)
# pca = PCA(n_components= 8)
# x2 = pca.fit_transform((X))
# pca_evr = pca.explained_variance_ratio_
# print(pca_evr)
=======
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

ds = load_diabetes()

X = ds.data
Y = ds.target

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.
print(cumsum)

print(np.argmax(cumsum >= 0.94) + 1)
# print(cumsum >= 0.94)
# pca = PCA(n_components= 8)
# x2 = pca.fit_transform((X))
# pca_evr = pca.explained_variance_ratio_
# print(pca_evr)
>>>>>>> 667c42ee521f20fb0ad8f218b4ec214b25aaf949
# print(sum(pca_evr))