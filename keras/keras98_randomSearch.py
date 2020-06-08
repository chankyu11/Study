import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense
from keras.layers import MaxPooling2D
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#데이터를 불러옴과 동시에 x, y와 트레인과 테스트를 분리해준다. 

# print(x_train.shape)#(60000, 28, 28)
# print(x_test.shape)#(10000, 28, 28)
# x_train = x_train.reshape(x_train.shape[0], 28,28,1).astype('float') / 255
# x_test = x_test.reshape(x_test.shape[0],28 ,28,1 ) / 255

x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float') / 255
x_test = x_test.reshape(x_test.shape[0],28 * 28) / 255


# print(x_train.shape)   # (60000, 28, 28, 1)
# print(x_test.shape)    # (10000, 28, 28, 1)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)

# 2. 모델

def build_model(drop = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (28 * 28, ), name = 'input')
    x = Dense(512, activation = 'relu', name = 'h1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'h2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'h3')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation = 'relu', name = 'h4')(x)
    x = Dropout(drop)(x)
    output = Dense(10, activation = 'softmax', name = 'output')(x)
    
    model = Model(inputs = inputs, outputs = output)
    model.compile(optimizer = optimizer, metrics = ['acc'], loss = 'categorical_crossentropy')
    return model

def create_hyperParameter():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout}

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
# 케라스 분류 모델을 사이킷런 형태로 싸겠습니다.

model = KerasClassifier(build_fn = build_model, verbose = 1)
hyperparameter = create_hyperParameter()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model, hyperparameter, cv = 3)
search.fit(x_train, y_train)

acc = search.score(x_test, y_test)
print(search.best_params_)
print(search.best_estimator_)

# y_pred = model.predict(x_test)
print("acc" , acc)
# print("최종 정답률: ", accuracy_score(y_test, y_pred))
'''
{'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 20}
{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}
'''
