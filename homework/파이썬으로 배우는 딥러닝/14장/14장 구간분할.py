import pandas as pd
from pandas import DataFrame

attri_data1 = {"ID": ["99","100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울","서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year" :[1995, 1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["늙은","영이", "순돌", "짱구", "태양", "션", "유리", "현아", "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)

birth_year_bins = [1980, 1985, 1990, 1995, 2000]

birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins)
# print(birth_year_cut_data)

# print(pd.value_counts(birth_year_cut_data))

group_names = ["first1980", "second1980", "first1990", "second1990"]
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins, labels = group_names)
# print(pd.value_counts(birth_year_cut_data))

# print(pd.cut(attri_data_frame1.birth_year, 2))
# print(pd.cut(attri_data_frame1.birth_year, 5))


attri_data1 = {"ID": ["99","100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울","서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year" :[1995, 1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["늙은","영이", "순돌", "짱구", "태양", "션", "유리", "현아", "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)

x = pd.cut(attri_data_frame1.ID,2)
print()