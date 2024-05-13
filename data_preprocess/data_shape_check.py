import os
import cv2
join = os.path.join

path = '/vast/AI_team/sukmin/datasets/TransferTest_NET/고영신'
path_list = sorted(next(os.walk(path))[2])

for case in path_list:
    c_path = join(path, case)
    img = cv2.imread(c_path)
    print(case)
    print(img.shape)
    print()