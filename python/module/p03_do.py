import p01_car
import p02_tv

print("===" * 20)

p01_car.drive()
p02_tv.watch()
''' 
결과가 아래와 같이 나오는 이유는 import하면서 위 파일 내용이 실행.
그리고 함수안에 내용을 다시 print해서 결과가 이럼.

운전하다
시청하다
============================================================
운전하다
시청하다 

'''