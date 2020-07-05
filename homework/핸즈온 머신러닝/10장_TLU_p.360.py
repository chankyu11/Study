## p. 361

## 사이킷런은 아나의 TLU 네트워크를 구현한 Perceptron 클래스를 제공한다. 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
x = iris.data[:, (2,3)]    # 꽃잎의 길이와 너비
y = (iris.target == 0).astype(np.int)  # 부채붓꽃인가?

per_clf = Perceptron()
per_clf.fit(x,y)

y_pred = per_clf.predict([[2,0.5]])

print(y_pred)

# 퍼셉트론 학습 알고리즘은 확률적 경사 하강법과 비슷.
# 사이킷런의 Perceptron은 매개변수가 loss = 'perceptron', learning_rate = "constant",
# eta0 = 1, penalty = None(규제없음)인 SGDlassifier과 같다.

