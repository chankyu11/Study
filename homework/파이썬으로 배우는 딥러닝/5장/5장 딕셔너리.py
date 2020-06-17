# p.125
dic ={"Japan": "Tokyo", "Korea": "Seoul"}
print(dic)

town = {"경기도": "수원", "서울": "중구"}
print(town)

# p.125 딕셔너리 요소 추출

dic ={"Japan": "Tokyo", "Korea": "Seoul"}
print(dic["Japan"])

# p.126

town = {"경기도": "수원", "서울": "중구"}
print("경기도의 중심 도시는", town["경기도"], "입니다.")
print("서울의 중심 도시는", town["서울"], "입니다.")


# 딕셔너리 갱신 및 추가

# 딕셔너리명[값을 갱신할 키] = 값

dic ={"Japan":"Tokyo","Korea":"Seoul"}
dic["Japan"] = "Osaka"
dic["China"] = "Beijing"
# 딕셔너리 원래 값 = 변경할 값
# print(dic)

# p.127
town = {"경기도": "수원", "서울": "중구"}
town["제주도"]  = "제주시"
town["경기도"] = "분당"
print(town)

# 딕셔너리 요소 삭제

dic ={"Japan": "Tokyo", "Korea": "Seoul", "China": "Beijing"}
del dic["China"]
print(dic)

town = {"경기도": "수원", "서울": "중구"}
del town["경기도"]

print(town)