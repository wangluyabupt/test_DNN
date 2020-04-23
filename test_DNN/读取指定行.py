import pandas as pd
import numpy as np
a = [
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [5, 5, 5, 5],
    [6, 6, 6, 6],
    [7,7,7,7],
    [8,8,8,8],
    [9,9,9,9],
    [10,10,10,10]
]
a = pd.DataFrame(a)
a.to_csv("a.csv", index=False,header=None)

def get_next(filename,batch_size,M,i):
    if batch_size*i>=M:
        i=0
    x = np.array(pd.read_csv(filename, skiprows=batch_size * i, nrows=batch_size, header=None))
    x = x.astype(np.float32)
    return x,i+1

M=10000
epoch=2
batch_size=100
i=0

# x,i=get_next('a.csv',batch_size,M,6)
# print(x,i)
for m in range(int(epoch*(M/batch_size))):
    x,i=get_next('x_data.csv',batch_size,M,i)
    print(x,i)



