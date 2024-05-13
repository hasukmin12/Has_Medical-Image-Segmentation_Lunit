import glob
import os, tempfile
import random
from requests import post
import torch
import wandb
import argparse as ap
import numpy as np
import nibabel as nib
import yaml
from tqdm import tqdm
from typing import Tuple

from monai.inferers import sliding_window_inference
from monai.data import *
from monai.transforms import *
from skimage import io, segmentation, morphology, measure, exposure
import time
import tifffile as tif
import cv2
import cv2

join = os.path.join


path = "/vast/AI_team/sukmin/datasets/Task505_Stroma_NDM_prep_200x_whole/imagesTs"
path_list = sorted(next(os.walk(path))[2])

# test_images = sorted(glob.glob(os.path.join(input_path, "*.png")))

gt_dir = "/vast/AI_team/sukmin/datasets/Task505_Stroma_NDM_prep_200x_whole/labelsTs"
aim_overlay_dir = '/vast/AI_team/sukmin/datasets/Task601_Stroma_Only_Epi_200x_whole_image/overlay_gt'
os.makedirs(aim_overlay_dir, exist_ok=True)

aim_labelsTs_dir = '/vast/AI_team/sukmin/datasets/Task601_Stroma_Only_Epi_200x_whole_image/labelsTs'
os.makedirs(aim_labelsTs_dir, exist_ok=True)


for case in path_list:
    if case != 'Thumbs.db':
        print(case)
        img_path = join(path, case)
        gt_path = join(gt_dir, case)

        img = cv2.imread(img_path)
        seg = cv2.imread(gt_path, -1)

        new_seg = np.zeros(seg.shape, dtype=np.uint8)
        rst = np.zeros(img.shape, dtype=np.uint8)

        
        for x in range(seg.shape[0]):
            for y in range(seg.shape[1]):
                if seg[x][y] == 0:
                    rst[x][y] = [255, 255, 255]  # back-ground

                elif seg[x][y] == 1: # BGR
                    new_seg[x][y] = 1
                    rst[x][y] = [149, 184, 135]  # BGR  - stroma (연두)
                elif seg[x][y] == 2:
                    new_seg[x][y] = 2
                    rst[x][y] = [0, 242, 255]  # epithelium (노랑)
                elif seg[x][y] == 3:
                    new_seg[x][y] = 3
                    rst[x][y] = [192, 135, 160]  # others (보라)


                elif seg[x][y] == 4:
                    new_seg[x][y] = 2
                    rst[x][y] = [0, 242, 255]  # epithelium (노랑)

                elif seg[x][y] == 5:
                    new_seg[x][y] = 2
                    rst[x][y] = [0, 242, 255]  # epithelium (노랑)

                elif seg[x][y] == 6:
                    new_seg[x][y] = 2
                    rst[x][y] = [0, 242, 255]  # epithelium (노랑)


        alpha = 0.6
        rst2 = cv2.addWeighted(img, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
        # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)

        cv2.imwrite(rst2, join(aim_overlay_dir, case))
        cv2.imwrite(new_seg, join(aim_labelsTs_dir, case))








gt_dir = "/vast/AI_team/sukmin/datasets/Task505_Stroma_NDM_prep_200x_whole/labelsTr"
# path = path_tr
path_list = sorted(next(os.walk(gt_dir))[2])

aim_labelsTr_dir = '/vast/AI_team/sukmin/datasets/Task601_Stroma_Only_Epi_200x_whole_image/labelsTr'
os.makedirs(aim_labelsTr_dir, exist_ok=True)

for case in path_list:
    if case != 'Thumbs.db':
        print(case)
        # img_path = join(path, case)
        gt_path = join(gt_dir, case)

        # img = cv2.imread(img_path)
        seg = cv2.imread(gt_path, -1)

        new_seg = np.zeros(seg.shape, dtype=np.uint8)
        # rst = np.zeros(seg.shape, dtype=np.uint8)

        
        for x in range(seg.shape[0]):
            for y in range(seg.shape[1]):
                if seg[x][y] == 1: # BGR
                    new_seg[x][y] = 1
                elif seg[x][y] == 2:
                    new_seg[x][y] = 2
                elif seg[x][y] == 3:
                    new_seg[x][y] = 3

                elif seg[x][y] == 4:
                    new_seg[x][y] = 2                 

                elif seg[x][y] == 5:
                    new_seg[x][y] = 2
                
                elif seg[x][y] == 6:
                    new_seg[x][y] = 2
          

        cv2.imwrite(new_seg, join(aim_labelsTr_dir, case))
