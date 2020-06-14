# p. 425

import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

# 일부 데이터를 일부러 누락시킵니다
sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA 

# print(sample_data_frame)


sample_data_frame.dropna()
# print(sample_data_frame)


# p.427

import numpy as np
from numpy import nan as NA
import pandas as pd

np.random.seed(0)
sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA

sample_data_frame[[0, 2]].dropna()



