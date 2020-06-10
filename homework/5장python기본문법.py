# p.116

fruits_name_1 = "사과"
fruits_num_1 = 2
fruits_name_2 = "귤"
fruits_num_2 = 10

fruits = [[fruits_name_1, fruits_num_1], [fruits_name_2, fruits_num_2]]
print(fruits)

# p.117

a = [1, 2, 3, 4]
print(a[1])
print(a[-2])

fruits = ["apple", 2, "orange", 4, "grape", 3, "banana", 1]

print(fruits[1]) # (fruits[-7])
print(fruits[7])

# p.118

a = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
print(a[1:5])
print(a[1:-5])
print(a[:5])
print(a[6:])
print(a[0:20])

# p.120

al = ["a", "b", "c", "d", "e"]
al[0] = "A"
al[1:3] = ["B", "C"]
print(al)

al = al + ["f"] # 모두 list에 추가
al += ["g","h"]
al.append("i")
print(al)

# p.121
# 리스트에서 요소 삭제
a1 = ["a", "b", "c", "d", "e"]
del a1[3:]
del a1[0]
print(a1)

# p. 125
# 리스트 주의점

alphabet = ["a", "b", "c"]
alphabet_copy = alphabet
alphabet_copy[0] = "A"
print(alphabet)
# a가 A로 바뀜

alphabet = ["a", "b", "c"]
alphabet_copy = alphabet[:]
alphabet_copy[0] = "A"
print(alphabet)
# 이렇게해야 복사
