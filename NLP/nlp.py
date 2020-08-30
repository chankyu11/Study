<<<<<<< HEAD
# 핸드온 머신러닝 p.629

import numpy as np
import keras

shakespeare_url = "http://homl.info/shakespeare"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_txt = f.read()

# print(shakespeare_txt)

'''
모든 글자를 정수로 인코딩해야함. tokenizer을 사용하는게 한 방법. 
먼저 이 클래스의 객체를 텍스트에 훈련. 텍스트에서 사용되는 모든 글자를 찾아 
각기 다른 글자 ID에 맵핑. 이 ID는 1부터 시작해 고유한 글자 개수까지 만들어짐.
'''
tokenizer = keras.preprocessing.text.Tokenizer(char_level= True)
tokenizer.fit_on_texts(shakespeare_txt)
'''
char_level = True는 단어수준이 아닌 글자 수준 인코딩을 만드는 것.
이 클래스는 기본적으로 텍스트를 소문자로 변환 그게 싫으면 lower = False로 지정.
'''
# print(tokenizer.texts_to_sequences(["First"]))
# First = [[20, 6, 9, 8, 3]]
# print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))

max_id = len(tokenizer.word_index)       # 고유 글자 개수
datasets_size = tokenizer.document_count # 전체 글자 개수

# print(max_id)
# print(datasets_size)

encoded = np.array(tokenizer.texts_to_sequences([shakespeare_txt])) - 1
=======
# 핸드온 머신러닝 p.629

import numpy as np
import keras

shakespeare_url = "http://homl.info/shakespeare"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_txt = f.read()

# print(shakespeare_txt)

'''
모든 글자를 정수로 인코딩해야함. tokenizer을 사용하는게 한 방법. 
먼저 이 클래스의 객체를 텍스트에 훈련. 텍스트에서 사용되는 모든 글자를 찾아 
각기 다른 글자 ID에 맵핑. 이 ID는 1부터 시작해 고유한 글자 개수까지 만들어짐.
'''
tokenizer = keras.preprocessing.text.Tokenizer(char_level= True)
tokenizer.fit_on_texts(shakespeare_txt)
'''
char_level = True는 단어수준이 아닌 글자 수준 인코딩을 만드는 것.
이 클래스는 기본적으로 텍스트를 소문자로 변환 그게 싫으면 lower = False로 지정.
'''
# print(tokenizer.texts_to_sequences(["First"]))
# First = [[20, 6, 9, 8, 3]]
# print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))

max_id = len(tokenizer.word_index)       # 고유 글자 개수
datasets_size = tokenizer.document_count # 전체 글자 개수

# print(max_id)
# print(datasets_size)

encoded = np.array(tokenizer.texts_to_sequences([shakespeare_txt])) - 1
>>>>>>> 667c42ee521f20fb0ad8f218b4ec214b25aaf949
# print(encoded)