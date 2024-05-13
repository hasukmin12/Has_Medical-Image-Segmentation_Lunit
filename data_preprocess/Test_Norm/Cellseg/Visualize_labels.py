#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:12:04 2022

convert instance labels to three class labels:
0: background
1: interior
2: boundary
@author: jma
"""

import os
join = os.path.join
import argparse

from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
from PIL import Image
import staintools
import cv2
join = os.path.join
# from StainNorm_Fuction import *

path = '/vast/AI_team/sukmin/datasets/Test_Norm/Cellseg/Test/labels'
path_list = sorted(next(os.walk(path))[2])
aim_path = '/vast/AI_team/sukmin/datasets/Test_Norm/Cellseg/Test/labels_vis'
os.makedirs(aim_path, exist_ok=True)

for case in path_list:
    print(case)
    img_data = np.array(Image.open(join(path, case))) * 100
    print(img_data.max())
    cv2.imwrite(join(aim_path, case), img_data)
                