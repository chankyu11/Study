# p.146

vege = "potato"
n = [4,5,2,7,6]

print(len(vege))
print(len(n))

# 매서드

alp = ["a", "b", "c", "d", "e"]
alp.append("f")
print(alp)

number = [1,4,5,2,3]

print(sorted(number))
number.sort()
print(number)

# 문자형 매서드(upper, count)

city = "Seoul"
print(city.upper())
print(city.count("o"))

animal = "elephant"

animal_big = animal.upper()
print(animal_big)
print(animal.count("e"))

# 문자형 매서드 (format)

print("나는{}에서 태어나 {}에서 유년기를 보냈습니다.".format("서울", "중구"))

fr = "바나나"
co = "노란색"

print("{}는 {}입니다.".format(fr, co))

# 리스트형 메서드

alp = ["a","b","c","d","e","f"]

print(alp.index("a"))
print(alp.count("a"))

n = [3,6,8,6,3,2,4,6]

print(n.index(2))
print(n.count(6))

# 리스트형 매서트(sort)

list = [1, 10, 2, 20]
list.sort()
print(list)

# reverse()
list = ["나", "가", "다", "라", "마"]
list.reverse()
# list.sort() # 한글도 정렬 가능
print(list)

n = [53, 26, 37, 69, 24, 2]

n.sort()
print(n)
n.reverse()
print(n)

