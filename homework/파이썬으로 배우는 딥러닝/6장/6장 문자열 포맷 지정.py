## p.176

pai = 3.141592
print("원주율은 %f" % pai)
print("원주율은 %.2f" % pai)

def bmi(height, weight):
    return weight / height**2

print("bmi는 %.2f입니다" % bmi(1.80, 88))

def check_character(object, character):
    return object.count(character)

print(check_character([1, 3, 4, 5, 6, 4, 3, 2, 1, 3, 3, 4, 3], 3))
print(check_character("asdgaoirnoiafvnwoeo", "d"))

def binary_search(numbers, target_number):
    low = 0
    high = len(numbers)
    
    while low <= high:
        middle = (low + high) // 2
        if numbers[middle] == target_number:
            print("{1}은(는) {0}번째에 있습니다".format(middle, target_number))
            
            break

        elif numbers[middle] < target_number:
            low = middle + 1
        else:
            high = middle - 1

numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13]

target_number = 11

binary_search(numbers, target_number)