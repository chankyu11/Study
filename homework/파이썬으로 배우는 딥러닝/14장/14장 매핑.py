import pandas as pd
from pandas import DataFrame

attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year" :[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["영이", "순돌", "짱구", "태양", "션", "유리", "현아", "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)
print(attri_data_frame1)


city_map = {"서울": "서울",
            "광주": "전라도",
            "부산": "경상도",
            "대전": "충정도"}
print(city_map)

# region 열 추가
attri_data_frame1["region"] = attri_data_frame1["city"].map(city_map)
print(attri_data_frame1)



attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year" :[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["영이", "순돌", "짱구", "태양", "션", "유리", "현아", "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)

WE_map = {"서울" : "중부",
          "광주" : "남부",
          "부산" : "남부",
          "대전" : "중부"}

attri_data_frame1["WE"] = attri_data_frame1["city"].map(WE_map)

print(attri_data_frame1)