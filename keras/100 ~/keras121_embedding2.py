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
""" 
[[ 0  0  2  3]
 [ 0  0  1  4]
 [ 1  5  6  7]
 [ 0  8  9 10]
 [11 12 13 14]
 [ 0  0  0 15]
 [ 0  0  0 16]
 [ 0  0 17 18]
 [ 0  0 19 20]
 [ 0  0  0 21]
 [ 0  0  2 22]
 [ 0  0  1 23]] 
 """
# pad_x = pad_sequences(x, padding = 'post')

'''
[[ 2  3  0  0]
 [ 1  4  0  0]
 [ 1  5  6  7]
 [ 8  9 10  0]
 [11 12 13 14]
 [15  0  0  0]
 [16  0  0  0]
 [17 18  0  0]
 [19 20  0  0]
 [21  0  0  0]
 [ 2 22  0  0]
 [ 1 23  0  0]]
'''
# pad_x = pad_sequences(x, padding = 'post', value = 1.0)
'''
[[ 2  3  1  1]
 [ 1  4  1  1]
 [ 1  5  6  7]
 [ 8  9 10  1]
 [11 12 13 14]
 [15  1  1  1]
 [16  1  1  1]
 [17 18  1  1]
 [19 20  1  1]
 [21  1  1  1]
 [ 2 22  1  1]
 [ 1 23  1  1]]
'''
print(pad_x)
# print(pad_x.shape)

word_size = len(token.word_index) + 1
print("전체 토큰 사이즈:", word_size)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
# model.add(Embedding(25, 10, input_length = 5))   # (none, 5, 10)
# Embedding은 두 번째 들어간다.
# 10은 노드 사이즈이다. 몇을 넣던지 상관없다.
# 통상적으로 단어의 수만큼 노드 사이즈를 넣어준다.
model.add(Embedding(25, 10))

# model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
#               metrics = ['acc'])
# model.fit(pad_x, labels, epochs = 30)

# acc = model.evaluate(pad_x, labels)[1]
# print("acc: ", acc)