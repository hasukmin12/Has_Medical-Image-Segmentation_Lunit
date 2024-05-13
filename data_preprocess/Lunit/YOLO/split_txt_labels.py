# re-read_img.py를 실행 해줘야만 라벨을 RGB로 인지한다
# https://github.com/open-mmlab/mmsegmentation/issues/550

import shutil
import os
import json
import tifffile as tif
# from skimage import io, segmentation, morphology, exposure
# from batchgenerators.utilities.file_and_folder_operations import *
join = os.path.join
import numpy as np


path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_for_YOLO/cell_detection_txt_r14_only1_start0"
path_list = sorted(next(os.walk(path))[2])
aim_path_t = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_for_YOLO_training_r14_only_cell_start0/train/labels"
aim_path_v = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_for_YOLO_training_r14_only_cell_start0/valid/labels"
# aim_path_t = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_for_YOLO/cell_detection_r14_start0/train/labels"
# aim_path_v = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_for_YOLO/cell_detection_r14_start0/valid/labels"
os.makedirs(aim_path_t, exist_ok=True)
os.makedirs(aim_path_v, exist_ok=True)

if "Thumbs.db" in path_list:
    path_list.remove("Thumbs.db") 

# train_list = path_list[80:]
# val_list = path_list[:80]

val_len = int(len(path_list) * 0.2)
train_patients = path_list[val_len:]
val_patients = path_list[:val_len]


for p in train_patients:
    if p != "Thumbs.db":
        print(p)
        input = join(path, p)
        output = join(aim_path_t, p)
        shutil.copy(input, output)



for p in val_patients:
    if p != "Thumbs.db":
        print(p)
        input = join(path, p)
        output = join(aim_path_v, p)
        shutil.copy(input, output)

