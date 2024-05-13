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
aim_dir = '/vast/AI_team/sukmin/datasets/Task505_Stroma_NDM_prep_200x_whole/overlay_gt'
os.makedirs(aim_dir, exist_ok=True)


for case in path_list:
    if case != 'Thumbs.db':
        print(case)
        img_path = join(path, case)
        gt_path = join(gt_dir, case)

        img = cv2.imread(img_path)
        seg = cv2.imread(gt_path)
        rst = np.zeros(img.shape, dtype=np.uint8)


        for x in range(seg.shape[0]):
            for y in range(seg.shape[1]):
                if seg[x][y][0] == 0:
                    rst[x][y] = [255, 255, 255]  # back-ground

                elif seg[x][y][0] == 1: # BGR
                    rst[x][y] = [149, 184, 135]  # BGR  - stroma (연두)
                elif seg[x][y][0] == 2:
                    rst[x][y] = [0, 242, 255]  # epithelium (노랑)
                elif seg[x][y][0] == 3:
                    rst[x][y] = [192, 135, 160]  # others (보라)
                elif seg[x][y][0] == 4:
                    rst[x][y] = [7, 153, 237]  # D (주황)
                elif seg[x][y][0] == 5:
                    rst[x][y] = [69, 48, 237]  # M (빨강)
                elif seg[x][y][0] == 6:
                    rst[x][y] = [232, 237, 4]  # NET (하늘)


        alpha = 0.6
        rst2 = cv2.addWeighted(img, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
        # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)

        cv2.imwrite(rst2, join(aim_dir, case))

