# Section12-1
# 파이썬 데이터 베이스 연동(SQLite)
# 테이블 생성 및 삽입

import sqlite3
import datetime

# 삽입 날짜 생성
now = datetime.datetime.now()
print("now: ", now)

# 시간보기 편하게
nowdatetime = now.strftime('%Y-%m-%d  %H:%M:%S')
print('nowdatetime', nowdatetime)

# sqlite3
print('sqlite3.version', sqlite3.version)
print('sqlite3._version', sqlite3.sqlite_version)

# DB생성 & Auto Commit <-> (Rollback)
conn = sqlite3.connect('D:/STUDY/fast_cam/SQL/database.db', isolation_level = None)

# Cursor

cursor = conn.cursor()
print('cursor Type: ', type(cursor))

# 테이블 생성(data type: TEXT, NUMERIC INTEGER, REAL BLOB)
cursor.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username text, email text, phone text, website text, regdate text)")

# 데이터 삽입

cursor.execute("INSERT INTO users VALUES(1, 'kim', 'lee@naver.com', '010-0012-1212', 'lee.com', ?)", (nowdatetime, ))
cursor.execute("INSERT INTO users(id, username, email, phone, website, regdate) VALUES(?,?,?,?,?,?)",(2, 'lee', 'park@naver.com', '010-2332-2332', 'google.com', nowdatetime))

# Many 삽입(튜플, 리스트)

userList = (
    (3, 'lee', 'lee@naver.com', '010-1233-1234', 'lee.com', nowdatetime),
    (4, 'cho', 'cho@naver.com', '010-1236-1234', 'cho.com', nowdatetime),
    (5, 'kwon', 'kwon@naver.com', '010-1239-1234', 'kwon.com', nowdatetime)
)

cursor.executemany("INSERT INTO users(id, username, email, phone, website, regdate) VALUES(?,?,?,?,?,?)", userList)

# 테이블 데이터 삭제

# conn.execute("DELETE FROM users")

# 테이블에 데이터 몇 개를 삭제 했는지 보여줌
# print("user db deleted: ", conn.execute("DELETE FROM users").rowcount)

# 커밋: isolation_level = None 일 경우 자동반영(오토 커밋)
# 이게 아니라면 conn.commit()을 사용해야함
# conn.rollback()을 사용하면 취소와 같은 개념

# 리소스를 사용할 때 conn.close()를 꼭 해야함
conn.close()

# Section12-1
# 파이썬 데이터 베이스 연동(SQLite)
# 테이블 생성 및 삽입

import sqlite3
import datetime

# 삽입 날짜 생성
now = datetime.datetime.now()
print("now: ", now)

# 시간보기 편하게
nowdatetime = now.strftime('%Y-%m-%d  %H:%M:%S')
print('nowdatetime', nowdatetime)

# sqlite3
print('sqlite3.version', sqlite3.version)
print('sqlite3._version', sqlite3.sqlite_version)

# DB생성 & Auto Commit <-> (Rollback)
conn = sqlite3.connect('D:/STUDY/fast_cam/SQL/database.db', isolation_level = None)

# Cursor

cursor = conn.cursor()
print('cursor Type: ', type(cursor))

# 테이블 생성(data type: TEXT, NUMERIC INTEGER, REAL BLOB)
cursor.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username text, email text, phone text, website text, regdate text)")

# 데이터 삽입

cursor.execute("INSERT INTO users VALUES(1, 'kim', 'lee@naver.com', '010-0012-1212', 'lee.com', ?)", (nowdatetime, ))
cursor.execute("INSERT INTO users(id, username, email, phone, website, regdate) VALUES(?,?,?,?,?,?)",(2, 'lee', 'park@naver.com', '010-2332-2332', 'google.com', nowdatetime))

# Many 삽입(튜플, 리스트)

userList = (
    (3, 'lee', 'lee@naver.com', '010-1233-1234', 'lee.com', nowdatetime),
    (4, 'cho', 'cho@naver.com', '010-1236-1234', 'cho.com', nowdatetime),
    (5, 'kwon', 'kwon@naver.com', '010-1239-1234', 'kwon.com', nowdatetime)
)

cursor.executemany("INSERT INTO users(id, username, email, phone, website, regdate) VALUES(?,?,?,?,?,?)", userList)

# 테이블 데이터 삭제

# conn.execute("DELETE FROM users")

# 테이블에 데이터 몇 개를 삭제 했는지 보여줌
# print("user db deleted: ", conn.execute("DELETE FROM users").rowcount)

# 커밋: isolation_level = None 일 경우 자동반영(오토 커밋)
# 이게 아니라면 conn.commit()을 사용해야함
# conn.rollback()을 사용하면 취소와 같은 개념

# 리소스를 사용할 때 conn.close()를 꼭 해야함
conn.close()

