## p.133
# animals = ["tiger", "dog", "elephant"]
# for animal in animals:
    # print(animal)


## p.134
# storages = [1,2,3,4]

# for i in storages:
    # print(i)

## break 
# storages = [1,2,3,4,5,6,7,8,9,10]
# for n in storages:
#     print(n)
#     if n >= 5:
#         print("끝")
#         break

## p.135
# storages = [1,2,3,4,5,6]

# for n in storages:
#     print(n)
#     if n == 4:
#         break;

## p.136 
## continue

# storages = [1,2,3]
# for n in storages:
#     if n ==2:
#         continue
#     print(n)

# storages = [1,2,3,4,5,6]

# for n in storages:
#     if n % 2 ==0:
#         continue
#     print(n)

## p.137
## for문에서 index표시

'''
for x, y in enumerate("리스트형"):
    for 안에서는 x, y를 사용하여 작성.
    x는 정수형의 인덱스, y는 리스트에 포함된 요소.
'''

# list = ["a", "b"]
# for index, value in enumerate(list):
#     print(index, value)

# l = ["tiger", "dog","elephant"]
# for index, value in enumerate(l):
#     print(str(index), value)

## 리스트 안의 리스트 루프

# l = [[1,2,3], [4,5,6]]
# for a,b,c in l:
#     print(a)    # 1,4
#     # print(b)  # 2,5
#     # print(c)  # 3,6

# f = [["strawberry", "red"],
#      ["peach", "pink"],
#      ["banan", "yellow"]]
# for a,b in f:
#     print(a,"is",b)
    
## 딕셔너리형의 루프

# f = {"strawberry": "red",
#     "peach": "pink",
#     "banana": "yellow"}
# for f , c in f.items():
#     print(f, "is", c)

town = {"경기도": "분당", "서울": "중구", "제주도": "제주시"}
for t, c in town.items():
    print(t,c)