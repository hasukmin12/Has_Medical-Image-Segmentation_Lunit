#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
join = os.path.join
import argparse
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2
import cv2


path = '/vast/AI_team/sukmin/datasets/2차annotation/40x(Downsample5)'
NDM_list = sorted(next(os.walk(path))[1])

case_num = 0
# N_list = []
# D_list = []
# M_list = []
# NET_list = []
wrong_list = []

for block in NDM_list:
        print()
        print()
        print(block, " is start")
        block_path = join(path, block)
        folder_list = sorted(next(os.walk(block_path))[1])

        for folder in folder_list:
            folder_path = join(block_path, folder)
            case_list = sorted(next(os.walk(folder_path))[2])

            # 여기서부터가 파일 접근
            for case in case_list:
                if case != 'Thumbs.db':

                    if case[-9:] != 'label.png':
                        img_path = join(folder_path, case)

                        image = cv2.imread(img_path)
                        width, height, _ = image.shape
                        if width < 256 or height < 256:
                             print(case_num)
                             print(case)
                             print()
                             wrong_list.append(case)



                        case_num += 1

print("wrong len : ", len(wrong_list))

