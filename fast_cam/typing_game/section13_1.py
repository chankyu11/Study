<<<<<<< HEAD
# section13-1
# 업그레이드 타이핑 게임 제작
# 타이핑 게임 제작 및 기본 완성

import random
import time

words = [] # 영어 단어 리스트 (1000개 로드)

n = 1     # 게임 시도 횟수
cor_cnt = 0  # 정답 개수

with open('D:/STUDY/fast_cam/typing_game/word.txt', 'r') as f:
    for c in f:
        words.append(c.strip())

# print(words) # 단어 리스트 확인

input("Ready? press enter key!") # enter game start!

start = time.time()

while n <= 5:
    random.shuffle(words)
    # 랜덤으로 섞어!
    q = random.choice(words)
    # 랜덤으로 뽑아온다 words에서

    print()
    
    print("*Question # {}".format(n))
    print(q)  # 여기서 문제 출력

    x = input() # 여기서 타이핑 받기!

    print()

    if str(q).strip() == str(x).strip(): #만약에 단어장에서 공백을 제거한 단어랑 == 입력 받은 내용의 공백을 제거한 내용이 같다면
        print("PASS!")
        cor_cnt += 1
    else:
        print("wrong!")
    
    n += 1  # 다음 문제로 !

end = time.time() # 끝!

et = end - start  # 총 게임시간
et = format(et, ".3f")  # 소숫라지 3자리

if cor_cnt >= 3:
    print("합격")

else:
    print("불합격")

# 수행 시간 출력
print("게임 시간: ", et, "초", "정답 개수 :{}".format(cor_cnt))

# 시작 지점 코드!
if __name__ == '__main__':
    pass
=======
# section13-1
# 업그레이드 타이핑 게임 제작
# 타이핑 게임 제작 및 기본 완성

import random
import time

words = [] # 영어 단어 리스트 (1000개 로드)

n = 1     # 게임 시도 횟수
cor_cnt = 0  # 정답 개수

with open('D:/STUDY/fast_cam/typing_game/word.txt', 'r') as f:
    for c in f:
        words.append(c.strip())

# print(words) # 단어 리스트 확인

input("Ready? press enter key!") # enter game start!

start = time.time()

while n <= 5:
    random.shuffle(words)
    # 랜덤으로 섞어!
    q = random.choice(words)
    # 랜덤으로 뽑아온다 words에서

    print()
    
    print("*Question # {}".format(n))
    print(q)  # 여기서 문제 출력

    x = input() # 여기서 타이핑 받기!

    print()

    if str(q).strip() == str(x).strip(): #만약에 단어장에서 공백을 제거한 단어랑 == 입력 받은 내용의 공백을 제거한 내용이 같다면
        print("PASS!")
        cor_cnt += 1
    else:
        print("wrong!")
    
    n += 1  # 다음 문제로 !

end = time.time() # 끝!

et = end - start  # 총 게임시간
et = format(et, ".3f")  # 소숫라지 3자리

if cor_cnt >= 3:
    print("합격")

else:
    print("불합격")

# 수행 시간 출력
print("게임 시간: ", et, "초", "정답 개수 :{}".format(cor_cnt))

# 시작 지점 코드!
if __name__ == '__main__':
    pass
>>>>>>> 667c42ee521f20fb0ad8f218b4ec214b25aaf949
