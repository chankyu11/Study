import pymssql as ms
# print('잘 됨')

conn = ms.connect(server = '127.0.0.1', user = 'bit2', password = '1234', database = 'bitdb', port = 1433)

cursor = conn.cursor()

cursor.execute("SELECT * FROM iris;")

row = cursor.fetchone()

while row :
    print("첫컬럼: %s, 둘컬럼: %s, 3컬럼: %s, 4컬럼: %s" %(row[0], row[1], row[2], row[4]))
    row = cursor.fetchone()

conn.close()
print("끝")
