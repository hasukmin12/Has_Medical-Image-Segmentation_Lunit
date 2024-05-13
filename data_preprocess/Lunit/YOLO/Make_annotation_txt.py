import os
import numpy as np
import shutil
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2
join = os.path.join

import csv


anno_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_Norm/annotations/train/cell'
anno_list = sorted(next(os.walk(anno_path))[2])

aim_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_for_YOLO/cell_detection_txt_r14_only1_start0'
os.makedirs(aim_path, exist_ok=True)

# output_path_for_visualize = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_for_YOLO/cell_detection_txt_visulization'
# os.makedirs(output_path_for_visualize, exist_ok=True)


for case in anno_list:
    if case != 'Thumbs.db':
        print(case)
        anno = open(join(anno_path, case), 'r', encoding='utf-8')
        pt = csv.reader(anno)
        seg = np.zeros((1024, 1024), dtype=np.uint8)

        # txt 파일 생성
        f=open(join(aim_path, "case_{0:03d}.txt".format(anno_list.index(case))), "w")

        for line in pt:
            # print(line)
            x = int(line[0])
            y = int(line[1])

            # make txt file
            # cls cx cy w h
            cls = 0 # int(line[2])
            cx = x/1024
            cy = y/1024
            
            # 반지름은 14으로 할 예정
            r = 14
            w = (2*r)/1024
            h = (2*r)/1024

            # txt write
            f.write("%d %f %f %f %f\n" %(cls,cx,cy,w,h))



        f.close()

    