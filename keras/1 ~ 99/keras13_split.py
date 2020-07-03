import numpy as np
# numpy를 불러오고 이제 np라고 칭하겠다.

# 1. 데이터
x = np.array(range(1,101))
# x에 넘파이의 어레이 형식으로 범위 1~100을 넣겠다.
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
# skleaen.model 패키지?? 클래스? 그 속에서 train_test_split를 불러와서 사용하겠다.

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = False, train_size = 0.8)
#   1       2        3       4
# x_train, x_test, y_train, y_test에 위에 선언 해놓은 x, y 를 대입하겠다. 
# shuffle = False, train_size = 0.8 shuffle의 기본 값은 True 따로 설정하지 않는다면 True
# train_size = 0.8 트레인 사이즈를 이미 선언 해놓은 x의 80%로 하겠다.
# 트레인 사이즈를 80%로 설정해놔서 1~80까지 들어가고, test에 나머지가 들어감.

# 2. 모델구성

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(10)) 
# model.add 레이어를 추가하겠다. 이번층부터 히든 레이어라고 부른다.
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 3. 훈련

model.compile(loss = 'mse', optimizer = adam, metrics = 'mse')
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split = 0.25)
# validation이 없어도 괜찮지만, 있으면 정확성을 높힐수 있음.
# validation은 train에 들어가 있어야함.
# 트레인, 벨리데이션, 테스트를 8 : 2 : 2로 맞추려한다. 
# validation_split = 0.25 = 위에 x_train을 이미 1~80으로 설정해놔서 80의 1/4인 0.25로 자른다.

# 4. 평가와 예측

loss, mse = model.evaluate(x_test, y_test, batch_size = 1)

print("loss:", loss)
print("mse:", mse)

# y_pred = model.predict(x_pred)
# print("y_pred:", y_pred)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

# R2구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)

