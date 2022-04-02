import pandas as pd
import numpy as np
import time
# Doc du lieu VCB 2009->2018
dataset_train = pd.read_csv('5000000 BT Records.csv')
training_set = dataset_train.iloc[:, 1:2].values

# print(training_set)

no_of_sample = len(training_set)
print(no_of_sample)

WINDOW_SIZE = 7 # 7 ngay
HORIZON_SIZE = 1 # 1 ngay

start = time.time()
steps = np.expand_dims(np.arange(WINDOW_SIZE + HORIZON_SIZE ),axis=0)
# print(steps, steps.shape)

add_matrix = np.expand_dims(np.arange(no_of_sample - WINDOW_SIZE - HORIZON_SIZE + 1), axis=0).T
# print(add_matrix, add_matrix.shape)

indexs = steps + add_matrix
# print(indexs, indexs.shape)

data  = training_set[indexs,0]
# print(data)

X_train, y_train = data[:,:WINDOW_SIZE], data[:,-HORIZON_SIZE:]
# print(X_train[0], y_train[0])
print("Time: ", time.time() - start)
# VCB 10 nam: Time:  0.00017690658569335938
# Bank transsaciton 1M: Time:  0.1764988899230957
# Bank transaction 5M: Time:  1.0231671333312988