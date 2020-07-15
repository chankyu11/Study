import pymssql as ms

conn = ms.connect(server = '127.0.0.1', user = 'bit2', password = '1234', database = 'bitdb', port = 1433)

cursor = conn.cursor()

cursor.execute("SELECT * FROM wine;")

row = cursor.fetchone()

# while row :
#     print("1컬럼: %s, 2컬럼: %s, 3컬럼: %s, 4컬럼: %s, 5컬럼: %s, 6컬럼: %s, 7컬럼: %s, 8컬럼: %s, 9컬럼: %s, 10컬럼: %s, 11컬럼: %s, 12컬럼: %s, 13컬럼: %s"%(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12]))
#     row = cursor.fetchone()

# conn.close()
# print("끝")
while row :
    for i in range(13):
        print("i컬럼: %s" %row[i])