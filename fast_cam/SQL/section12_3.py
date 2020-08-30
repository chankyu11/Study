<<<<<<< HEAD
# section12_3
# 파이썬 데이터베이스 연동(SQlite)
# 테이블 데이터 수정 및 삭제

import sqlite3

# DB생성
conn = sqlite3.connect('D:/STUDY/fast_cam/SQL/database.db')

c = conn.cursor()

# 데이터 수정1
# c.execute("UPDATE users SET username = ? WHERE id = ?",('niceman', 3))

# 데이터 수정2
# c.execute("UPDATE users SET username = :name WHERE id = :id",{'name':'goodman', 'id': '5'})

# 데이터 수정3
# c.execute("UPDATE users SET username = '%s' WHERE id = '%s'"%('badboy', 3))

# 중간 데이터 확인1
# for user in c.execute("SELECT * FROM users"):
#     print(user)

# Row delete1
# c.execute("DELETE FROM users WHERE id = ?", (2,))

# Row delete2
# c.execute("DELETE FROM users WHERE id = :id",{"id": 5})

# Row delete3
# c.execute("DELETE FROM users WHERE id = '%s'" %4)

# 테이블 전체 데이터 삭제

print("users db deleted:", c.execute("DELETE FROM users").rowcount, "rows")

# 커밋
conn.commit()

# 접속 해제
conn.close()

=======
# section12_3
# 파이썬 데이터베이스 연동(SQlite)
# 테이블 데이터 수정 및 삭제

import sqlite3

# DB생성
conn = sqlite3.connect('D:/STUDY/fast_cam/SQL/database.db')

c = conn.cursor()

# 데이터 수정1
# c.execute("UPDATE users SET username = ? WHERE id = ?",('niceman', 3))

# 데이터 수정2
# c.execute("UPDATE users SET username = :name WHERE id = :id",{'name':'goodman', 'id': '5'})

# 데이터 수정3
# c.execute("UPDATE users SET username = '%s' WHERE id = '%s'"%('badboy', 3))

# 중간 데이터 확인1
# for user in c.execute("SELECT * FROM users"):
#     print(user)

# Row delete1
# c.execute("DELETE FROM users WHERE id = ?", (2,))

# Row delete2
# c.execute("DELETE FROM users WHERE id = :id",{"id": 5})

# Row delete3
# c.execute("DELETE FROM users WHERE id = '%s'" %4)

# 테이블 전체 데이터 삭제

print("users db deleted:", c.execute("DELETE FROM users").rowcount, "rows")

# 커밋
conn.commit()

# 접속 해제
conn.close()

>>>>>>> 667c42ee521f20fb0ad8f218b4ec214b25aaf949
# DB는 여러 사람이 사용하고 엄청난 데이터를 손쉽게 사용.