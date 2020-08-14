BEGIN TRANSACTION;
CREATE TABLE users(id INTEGER PRIMARY KEY, username text, email text, phone text, website text, regdate text);
INSERT INTO "users" VALUES(1,'kim','lee@naver.com','010-0012-1212','lee.com','2020-08-13  15:08:17');
INSERT INTO "users" VALUES(2,'lee','park@naver.com','010-2332-2332','google.com','2020-08-13  15:08:17');
INSERT INTO "users" VALUES(3,'lee','lee@naver.com','010-1233-1234','lee.com','2020-08-13  15:08:17');
INSERT INTO "users" VALUES(4,'cho','cho@naver.com','010-1236-1234','cho.com','2020-08-13  15:08:17');
INSERT INTO "users" VALUES(5,'kwon','kwon@naver.com','010-1239-1234','kwon.com','2020-08-13  15:08:17');
COMMIT;
