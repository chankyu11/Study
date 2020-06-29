## p.191
import numpy as np

arr = np.arange(10)
# print(arr)

arr = np.arange(10)
arr[0:4] = 1, 2, 3, 4
# print(arr)

## 요소 중 3,4,5만 출력

arr = np.arange(10)
# print(arr[3:6])

arr[3:6] = 24
# print(arr)

## ndarray 사용시 주의사항

arr1 = np.array([1, 2, 3, 4, 5])
# print(arr1)

arr2 = arr1
arr2[0] = 100
# print(arr1)

arr1 = np.array([1, 2, 3, 4, 5])
# print(arr1)

arr2 = arr1.copy()
arr2[0] = 100
# print(arr1)

## view와 copy
List = [x for x in range(10)]
print("리스트 형 데이터입니다")
print("list :", List)
print()

List_copy = List[:]
List_copy[0] = 100

print("리스트의 슬라이스는 복사본이 생성되므로, arr_List에는 arr_List_copy의 변경점이 반영되지 않습니다.")
print("arr_List:",List)
print()

arr_NumPy = np.arange(10)
print("NumPy의 ndarray 데이터입니다")
print("arr_NumPy:",arr_NumPy)
print()

arr_NumPy_view = arr_NumPy[:]
arr_NumPy_view[0] = 100

print("NumPy의 슬라이스는 view(데이터가 저장된 위치의 정보)가 대입되므로, arr_NumPy_view를 변경하면 arr_NumPy에 반영됩니다")
print("arr_NumPy:",arr_NumPy)
print()

# NumPy의 ndarray에서 copy()를 이용한 경우를 확인합니다
arr_NumPy = np.arange(10)
print("NumPy의 ndarray에서 copy()를 이용한 경우입니다")
print("arr_NumPy:",arr_NumPy)
print()

arr_NumPy_copy = arr_NumPy[:].copy()
arr_NumPy_copy[0] = 100
print("copy()를 사용하면 복사본이 생성되기 때문에 arr_NumPy_copy는 arr_NumPy에 영향을 미치지 않습니다")
print("arr_NumPy:",arr_NumPy)
