from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
# 이미지 가져오오오기이이이

img_dog = load_img('./data/dog_cat/dog1.jpg', target_size = (224,224))
img_dog1 = load_img('./data/dog_cat/dog2.jpg', target_size = (224,224))
img_cat = load_img('./data/dog_cat/cat1.jpg', target_size = (224,224))
img_suit = load_img('./data/dog_cat/suit.jpg', target_size = (224,224))
img_onion = load_img('./data/dog_cat/onion.jpg', target_size = (224,224))

plt.imshow(img_cat)
# plt.show()

from keras.preprocessing.image import img_to_array
# 이미지를 numpy로 변환 

arr_dog = img_to_array(img_dog)
arr_dog1 = img_to_array(img_dog1)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_onion = img_to_array(img_onion)

# print(arr_dog)
# print(type(arr_dog))
# print(arr_dog.shape)


from keras.applications.vgg16 import preprocess_input
# VGG16에서 원하는 모양으로 변환. RBG -> BGR 

arr_dog = preprocess_input(arr_dog)
arr_dog1 = preprocess_input(arr_dog1)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_onion = preprocess_input(arr_onion)

plt.imshow(arr_cat)
# plt.show()
# print(arr_dog)

# 이미지 데이터를 하나로 합친다.
import numpy as np

arr_input = np.stack([arr_dog, arr_dog1, arr_cat, arr_suit, arr_onion])
# print(arr_input.shape)

# 모델

model = VGG16()
probs = model.predict(arr_input)

print(probs)
print("probs: ", probs.shape)

# 이미지 결과

from keras.applications.vgg16 import decode_predictions

result = decode_predictions(probs)

print("===" * 10)
print(result[0])
print("===" * 10)
print(result[1])
print("===" * 10)
print(result[2])
print("===" * 10)
print(result[3])
print("===" * 10)
print(result[4])
