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




path = "/vast/AI_team/sukmin/datasets/Task200_Stroma/imagesTs"
path_list = sorted(next(os.walk(path))[2])

# test_images = sorted(glob.glob(os.path.join(input_path, "*.png")))

gt_dir = "/vast/AI_team/sukmin/datasets/Task200_Stroma/labelsTs"
aim_dir = '/vast/AI_team/sukmin/datasets/Task200_Stroma/overlay_gt'
os.makedirs(aim_dir, exist_ok=True)


for case in path_list:
    print(case)
    img_path = join(path, case)
    gt_path = join(gt_dir, case)

    img = cv2.imread(img_path)
    seg = cv2.imread(gt_path)


    for x in range(seg.shape[0]):
        for y in range(seg.shape[1]):
            if seg[x][y][0] == 0:
                seg[x][y] = [255, 255, 255]  # back-ground

            elif seg[x][y][0] == 1:
                seg[x][y] = [133, 133, 218]  # BGR  - stroma (핑크)
            elif seg[x][y][0] == 2:
                seg[x][y] = [149, 184, 135]  # epithelium (연두)
            elif seg[x][y][0] == 3:
                seg[x][y] = [192, 135, 160]  # others (보라)


    alpha = 0.6
    rst = cv2.addWeighted(img, 1 - alpha, seg, alpha, 0.0, dtype = cv2.CV_32F)
    # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)

    cv2.imwrite(rst, join(aim_dir, case))

