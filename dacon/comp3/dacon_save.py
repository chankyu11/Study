
import numpy as np
import pandas as pd

test_features = pd.read_csv('./dacon/comp3/test_features.csv', header = 0, index_col = 0, encoding = 'cp949')
train_features = pd.read_csv('./dacon/comp3/train_features.csv', header = 0, index_col = 0, encoding = 'cp949')
train_target = pd.read_csv('./dacon/comp3/train_target.csv', header = 0, index_col = 0, encoding = 'cp949')
sample = pd.read_csv('./dacon/comp3/sample_submission.csv', header = 0, index_col = 0, encoding = 'cp949')

test_features = test_features.drop('Time', axis = 1)
train_features = train_features.drop('Time', axis = 1)
# Time 라인 삭제

# train_features1 = train_features.groupby(['id']).mean()
# test_features1 = test_features.groupby(['id']).mean()
# train_target1 = train_target.groupby(['id']).mean()

# print(test_featuers)
# print(train_featuers)

# train_featuers.to_csv('./dacon/comp3/train_featuers_group.csv', mode = 'w')

test_features2 = test_features.values
train_features2 = train_features.values
train_target2 = train_target.values
sample2 = sample.values

np.save('./dacon/comp3/test_f1.npy', arr = test_features2)
np.save('./dacon/comp3/train_f1.npy', arr = train_features2)
np.save('./dacon/comp3/train_t1.npy', arr = train_target2)
np.save('./dacon/comp3/sample1.npy', arr = sample2)

# import numpy as np
# import pandas as pd

# test_features = pd.read_csv('./dacon/comp3/test_features.csv', header = 0, index_col = 0, encoding = 'cp949')
# train_features = pd.read_csv('./dacon/comp3/train_features.csv', header = 0, index_col = 0, encoding = 'cp949')
# train_target = pd.read_csv('./dacon/comp3/train_target.csv', header = 0, index_col = 0, encoding = 'cp949')
# sample = pd.read_csv('./dacon/comp3/sample_submission.csv', header = 0, index_col = 0, encoding = 'cp949')

# test_features = test_features.drop('Time', axis = 1)
# train_features = train_features.drop('Time', axis = 1)
# # Time 라인 삭제

# train_features1 = train_features.groupby(['id']).mean()
# test_features1 = test_features.groupby(['id']).mean()
# train_target1 = train_target.groupby(['id']).mean()

# # print(test_featuers)
# # print(train_featuers)

# # train_featuers.to_csv('./dacon/comp3/train_featuers_group.csv', mode = 'w')

# test_features2 = test_features1.values
# train_features2 = train_features1.values
# train_target2 = train_target1.values
# sample2 = sample.values

# np.save('./dacon/comp3/test_f.npy', arr = test_features2)
# np.save('./dacon/comp3/train_f.npy', arr = train_features2)
# np.save('./dacon/comp3/train_t.npy', arr = train_target2)
# np.save('./dacon/comp3/sample.npy', arr = sample2)


