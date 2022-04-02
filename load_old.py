import pandas as pd
import numpy as np
import time
# Doc du lieu VCB 2009->2018
dataset_train = pd.read_csv('5000000 BT Records.csv')
training_set = dataset_train.iloc[:, 1:2].values

print(training_set)

# Tao du lieu train, X = 60 time steps, Y =  1 time step
X_train = []
y_train = []
no_of_sample = len(training_set)

WINDOW_SIZE = 7 # 7 ngay
HORIZON_SIZE = 1 # 1 ngay

start = time.time()
for i in range(WINDOW_SIZE, no_of_sample):
    X_train.append(training_set[i-WINDOW_SIZE:i, 0])
    y_train.append(training_set[i:i+HORIZON_SIZE, 0])

print("Time: ", time.time() - start)
# VCB 10 nam:  0.002601146697998047
# 1M ban ghi Bank Transaction: Time:  1.127094030380249
# 5M ban ghi BT: Time:  6.83643913269043

# VCB 10 nam: Time:  0.00017690658569335938
# Bank transsaciton 1M: Time:  0.1764988899230957
# Bank transaction 5M: Time:  1.0231671333312988

X_train, y_train = np.array(X_train), np.array(y_train)