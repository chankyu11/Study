<<<<<<< HEAD
# Section12-2
# 파이썬 데이터 베이스 연동(SQLite)
# 테이블 조회

import sqlite3

# DB파일 조회(없으면 새로 생성)

conn = sqlite3.connect('D:/STUDY/fast_cam/SQL/database.db') # 본인 DB 경로

# 커서 바인딩

c = conn.cursor()

# 데이터 조회(전체)
c.execute("SELECT * FROM users")

# 커서 위치 변경
# 1개 로우 선택

# print('one -> \n', c.fetchone())

# 지정 로우 선택
# print('Three -> \n', c.fetchmany(3))

# 전체 로우 선택
# print('ALL -> \n', c.fetchall())

# 커서가 위치를 기억하고 있다. 그래서 하나 뽑아오고 세개 뽑을대 2~4까지 출력되고 다 뽑으라니까 '5'만 나온거

print("===" * 30)

# 순회1

# rows = c.fetchall()
# for row in rows:
#     print('retrieve1 >', row)

# 순회2
# for row in c.fetchall():
#     print('retrieve2 >', row)

# # 순회3
# for row in c.execute('SELECT * FROM users ORDER BY id desc'):
#     print('retrieve3 >', row)
# 코드를 이렇게 작성하면 c.execute를 따로 안해줘도 됨
print("===" * 30)

# WHERE Retrieve1

param1 = (3, )
c.execute('SELECT * FROM users WHERE id = ?', param1)
print('param1', c.fetchone())
print('param1', c.fetchall())

# WHERE Retrieve2

param2 = 4
c.execute('SELECT * FROM users WHERE id = "%s"' %param2) # %s 문자 %f소수 %d정수
print('param2', c.fetchone())
print('param2', c.fetchall())

# WHERE Retrieve3

c.execute('SELECT * FROM users WHERE id = :Id' ,{"Id" : 5}) # %s 문자 %f소수 %d정수
print('param3', c.fetchone())
print('param3', c.fetchall())

# WHERE Retrieve4

param4 = (3,5)
c.execute('SELECT * FROM users WHERE id IN(?,?)', param4)
print('param4:', c.fetchall())
# 위와 방식은 비슷하게 따로 param선언 no
c.execute("SELECT * FROM users WHERE id IN('%d','%d')"%(3,4))
print('param4:', c.fetchall())

# WHERE Retrieve4
# or 연산자로 조회
c.execute("SELECT * FROM users WHERE id= :id1 OR id=:id2",{"id1":2, "id2":5})
print('param4:', c.fetchall())

# dump 출력

with conn:
    with open("D:/STUDY/fast_cam/SQL/dump.sql",'w') as f:
        for line in conn.iterdump():
            f.write('%s\n' % line)
        print('Dump print complete')

# f.close(), conn.close() 가 with문 때문에 자동으로 생성 
=======
# Section12-2
# 파이썬 데이터 베이스 연동(SQLite)
# 테이블 조회

import sqlite3

# DB파일 조회(없으면 새로 생성)

conn = sqlite3.connect('D:/STUDY/fast_cam/SQL/database.db') # 본인 DB 경로

# 커서 바인딩

c = conn.cursor()

# 데이터 조회(전체)
c.execute("SELECT * FROM users")

# 커서 위치 변경
# 1개 로우 선택

# print('one -> \n', c.fetchone())

# 지정 로우 선택
# print('Three -> \n', c.fetchmany(3))

# 전체 로우 선택
# print('ALL -> \n', c.fetchall())

# 커서가 위치를 기억하고 있다. 그래서 하나 뽑아오고 세개 뽑을대 2~4까지 출력되고 다 뽑으라니까 '5'만 나온거

print("===" * 30)

# 순회1

# rows = c.fetchall()
# for row in rows:
#     print('retrieve1 >', row)

# 순회2
# for row in c.fetchall():
#     print('retrieve2 >', row)

# # 순회3
# for row in c.execute('SELECT * FROM users ORDER BY id desc'):
#     print('retrieve3 >', row)
# 코드를 이렇게 작성하면 c.execute를 따로 안해줘도 됨
print("===" * 30)

# WHERE Retrieve1

param1 = (3, )
c.execute('SELECT * FROM users WHERE id = ?', param1)
print('param1', c.fetchone())
print('param1', c.fetchall())

# WHERE Retrieve2

param2 = 4
c.execute('SELECT * FROM users WHERE id = "%s"' %param2) # %s 문자 %f소수 %d정수
print('param2', c.fetchone())
print('param2', c.fetchall())

# WHERE Retrieve3

c.execute('SELECT * FROM users WHERE id = :Id' ,{"Id" : 5}) # %s 문자 %f소수 %d정수
print('param3', c.fetchone())
print('param3', c.fetchall())

# WHERE Retrieve4

param4 = (3,5)
c.execute('SELECT * FROM users WHERE id IN(?,?)', param4)
print('param4:', c.fetchall())
# 위와 방식은 비슷하게 따로 param선언 no
c.execute("SELECT * FROM users WHERE id IN('%d','%d')"%(3,4))
print('param4:', c.fetchall())

# WHERE Retrieve4
# or 연산자로 조회
c.execute("SELECT * FROM users WHERE id= :id1 OR id=:id2",{"id1":2, "id2":5})
print('param4:', c.fetchall())

# dump 출력

with conn:
    with open("D:/STUDY/fast_cam/SQL/dump.sql",'w') as f:
        for line in conn.iterdump():
            f.write('%s\n' % line)
        print('Dump print complete')

# f.close(), conn.close() 가 with문 때문에 자동으로 생성 
>>>>>>> 667c42ee521f20fb0ad8f218b4ec214b25aaf949
# dump를 통해 