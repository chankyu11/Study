# keras122_embedding를 카피 conv1d로 구성
from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "참 최고에요", "참 잘 만든 영화네요",
        '추천하고 싶은 영화입니다', ' 한 번 더 보고 싶네요', '글쌔요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', ' 참 재밌네요']

# 긍정1. 부정0

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding = 'pre')
# pad sequences를 사용하여 shape 맞추기

print(pad_x)
# print(pad_x.shape)

word_size = len(token.word_index) + 1
print("전체 토큰 사이즈:", word_size)

# labels = labels.reshape(12, 1, 1)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout

model = Sequential()
model.add(Embedding(word_size, 10, input_length = 5))   # (none, 5, 10)
# Embedding은 두 번째 들어간다.
# 10은 노드 사이즈이다. 몇을 넣던지 상관없다.
# 통상적으로 단어의 수만큼 노드 사이즈를 넣어준다.
# model.add(Embedding(25, 10))
model.add(Conv1D(1000, 1, padding = 'same', activation = 'relu'))
# embedding과 LSTM을 같이 사용할때는 input_length를 사용할 필요없다.
# embedding은 3차원
# 자연어에서 LSTM을 가장 많이 사용함.

model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

'=================================================================='

# model = Sequential()
# model.add(Embedding(25, 10, input_length=5))
# model.add(Conv1D(16, 3,
#                  padding='valid',
#                  activation='relu'))
# model.add(MaxPooling1D())
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model.summary()


# 
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics = ['acc'])
model.fit(pad_x, labels, epochs = 30)

acc = model.evaluate(pad_x, labels)[1]
print("acc: ", acc)