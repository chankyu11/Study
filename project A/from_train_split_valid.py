<<<<<<< HEAD
import random

count = 0
length = 660 #total line

txt = open('D:/darknet-master/build/darknet/x64/data/data/train_suffle.txt','r')

i = 0

f = open('D:/darknet-master/build/darknet/x64/data/data/train1.txt','w')
f2 = open('D:/darknet-master/build/darknet/x64/data/data/valid.txt','w')

while True :
    if i == 0 :
        line = txt.readline()
        if not line :
            break
        count +=1
        if count < int(length/10)*2 :
            f2.write(line)
        else :
            f.write(line)

txt.close()
f.close()
=======
import random

count = 0
length = 660 #total line

txt = open('D:/darknet-master/build/darknet/x64/data/data/train.txt','r')

i = 0

f = open('D:/darknet-master/build/darknet/x64/data/data/train1.txt','w')
f2 = open('D:/darknet-master/build/darknet/x64/data/data/valid.txt','w')

while True :
    if i == 0 :
        line = txt.readline()
        if not line :
            break
        count +=1
        if count < int(length/10)*2 :
            f2.write(line)
        else :
            f.write(line)

txt.close()
f.close()
>>>>>>> 667c42ee521f20fb0ad8f218b4ec214b25aaf949
f2.close()