from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size =0.8, random_state= 42
)

model = RandomForestClassifier()

# max_feature: 기본 값 사용
# n_estimators: 클수록 좋다, 단점 메모리 엄청 차지, 기본 값 100
# n_jobs: 병렬처리  -1 은 최대치

model.fit(x_train, y_train)

acc = model.score(x_test , y_test)

print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align = 'center')
    # bar는 세로 막대, barh 가로 막대
    plt.yticks(np.arange(n_features), cancer.feature_names)
    # y축의 정보
    
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    # ylim = y축 범위

plot_feature_importances_cancer(model)
plt.show()
