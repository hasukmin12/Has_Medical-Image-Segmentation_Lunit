import shutil
import os
import cv2
join = os.path.join

path = '/vast/AI_team/sukmin/datasets/Test_Norm/Cellseg/Test/Test_prep_Not_norm/labels'
path_list = sorted(next(os.walk(path))[2])
aim_path = '/vast/AI_team/sukmin/datasets/Test_Norm/Cellseg/Test/labels'
os.makedirs(aim_path, exist_ok=True)

for case in path_list:
    input = join(path, case)
    output = join(aim_path, "case_"+case[-7:])
    shutil.copy(input, output)