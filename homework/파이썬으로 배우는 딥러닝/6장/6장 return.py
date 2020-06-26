# p.161

def introduce(first = "김", second = "길동"):
    return "성은 " + first + "이고, 이름은 " + second + "입니다."

print(introduce("홍"))

def introduce(first = "김", second = "길동"):
    comment = "성은 " + first + "이고, 이름은 " + second + "입니다."
    return comment

print(introduce("홍"))

# bmi을 계산하는 함수, bmi를 반환 값으로 ㄱㄱ
def bmi(height, weight):
    return weight / height**2

print(bmi(1.80, 88))

