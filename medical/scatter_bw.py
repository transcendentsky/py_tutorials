import os
import sys
import numpy as np


scatter_list = [[0,1,2,3,4],[0,1,2,3,4]]
scatter_list = np.array(scatter_list)

matrix = np.zeros((4,4,1))
lists = [[] for i in range(25)] 
weight_lists = [0.0 for i in range(25)]
sort_lists = [[] for i in range(25)]
result_lists = [[] for i in range(25)]

# print(all_list[:3])
# Cluttering    
for i, point in enumerate(scatter_list):
    u = point[0] / 64.0 
    v = point[1] / 64.0 
    x = point[2] / 64.0
    y = point[3] / 64.0

    nu = round(u)
    du = u - n
    nv = round(v)
    dv = v - n

    distance = du * du + dv * dv
    weight = 1 / distance

    lists[ny*4+nx].append(list([i, x, y, weight]))
    sort_lists[ny*4+nx].append(weight)
    weights_list[ny*4+nx] += weight

def take_weight(elem):
    return elem[3]


# Caculate the grid points
for i, _list in enumerate(lists):
    # sort the weights list 
    sort_lists[i].sort(reverse=True)
    # 25 sublist
    for j, item in enumerate(sort_lists[i].items()):
        # x subsublist
        _i = item[0]
        _x = item[1]
        _y = item[2]
        _weight = item[3] / weights_list[i]

        x = _weight * _X
        y = _weight * _y

        result_lists[i].append(x)
        result_lists[i].append(y)




    





