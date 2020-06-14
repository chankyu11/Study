# p. 419

# import csv

# # with 문을 사용해 파일을 처리
# with open("csv0.csv", "w") as csvfile:
#     # writer() 메서드의 인수로 csvfile과 개행(줄바꿈) 코드(\n)를 지정합니다
#     writer = csv.writer(csvfile, lineterminator="\n")

#     # writerow(리스트) 로 행을 추가합니다
#     writer.writerow(["city", "year", "season"])
#     writer.writerow(["Nagano", 1998, "winter"])
#     writer.writerow(["Sydney", 2000, "summer"])
#     writer.writerow(["Salt Lake City", 2002, "winter"])
#     writer.writerow(["Athens", 2004, "summer"])
#     writer.writerow(["Torino", 2006, "winter"])
#     writer.writerow(["Beijing", 2008, "summer"])
#     writer.writerow(["Vancouver", 2010, "winter"])
#     writer.writerow(["London", 2012, "summer"])
#     writer.writerow(["Sochi", 2014, "winter"])
#     writer.writerow(["Rio de Janeiro", 2016, "summer"])

import csv
# 여기에 해답을 기술하세요

with open("ck.csv","w") as csvfile:
    writer = csv.writer(csvfile, lineterminator = "\n")

    writer.writerow(["birthday", "sex", "name"])
    writer.writerow(["921130", "male", "Lee Chankyu"])


