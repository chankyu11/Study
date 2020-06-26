# # p.155

# def sing():
#     print("노래합니다!")
    
# sing()

# def introduce():
#     print("홍길동입니다")

# introduce()

# ## 인수

# def introduce(n):
#     print(n + "입니다")

# introduce("홍길동")

## 인수를 세제곱한 값을 출력 함수

# def cube_cal(n):
#     print(n**3)

# print(cube_cal(3))

# ## 여러개의 인수

# def introduce(first, second):
#     print("성은", first, "이름은", second)

# print(introduce("이","찬규"))


# def introduce(first, second):
#     print(first + "입니다.", second, "살입니다.")

# print(introduce("홍길동","18"))

# ## 인수의 초깃값

# def introduce(first = "김", second = "길동"):
#     print("성은 " + first + "이고, 이름은 " + second + "입니다.")

# introduce("홍")

# def introduce(first, second = "길동"):
#     print("성은 " + first + "이고, 이름은 " + second + "입니다.")
# # 이렇게 함수에 인수를 넣어주면 오류 발생 ㄴㄴ, 안 넣으면 ㅇㅇ
# introduce("홍")

# def introduce(first = "홍", second):
#     print("성은 " + first + "이고, 이름은 " + second + "입니다.")
# # 이렇게 넣어도 오류

# # n은 홍길동 초깃값 설정, 나이는 18로 인수로 호출
# def introduce(age, n = "홍길동"):
#     print(n + "입니다. " + str(age) + "살입니다.")

# introduce(18)








