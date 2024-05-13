import numpy as np
import os
import nibabel as nib
import shutil
import cv2


path = '/home/sukmin/datasets/Task080_NeurIPS_Cellseg/labelsTr'
path_list = next(os.walk(path))[2]
path_list.sort()
print(path_list)
print()

list_u = []
list_b = []
list_k = []

u = 0
b = 0
total = 0
for case in path_list:
    print()
    print(case)
    case_path = os.path.join(path, case)
    seg = cv2.imread(case_path)
    z_axis = int(seg.shape[2])
    x_axis = int(seg.shape[0])
    y_axis = int(seg.shape[1])


    for x in range(x_axis):
        for y in range(y_axis):
            if seg[x][y][0] == 1:
                u += 1
            elif seg[x][y][0] == 2:
                b += 1

    total = u + b
    print(u/total)
    print(b/total)

    print((total / u) / (total / u + total / b))
    print((total / b) / (total / u + total / b))

print()
print(u/total)
print(b/total)

print()
print(total/u)
print(total/b)

print()
print((total/u) / (total/u + total/b))
print((total / b) / (total / u + total / b  ))
