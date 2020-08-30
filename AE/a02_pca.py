<<<<<<< HEAD
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

ds = load_diabetes()

X = ds.data
Y = ds.target

pca = PCA(n_components= 8)
x2 = pca.fit_transform((X))
pca_evr = pca.explained_variance_ratio_
print(pca_evr)
=======
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

ds = load_diabetes()

X = ds.data
Y = ds.target

pca = PCA(n_components= 8)
x2 = pca.fit_transform((X))
pca_evr = pca.explained_variance_ratio_
print(pca_evr)
>>>>>>> 667c42ee521f20fb0ad8f218b4ec214b25aaf949
print(sum(pca_evr))