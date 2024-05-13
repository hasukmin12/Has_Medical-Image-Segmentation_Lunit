#%%
import os, tempfile
import random
from requests import post
import torch
import wandb
import argparse as ap
import numpy as np
import nibabel as nib
import time
import cv2
import cv2
import matplotlib.pyplot as plt
join = os.path.join

path = '/vast/AI_team/sukmin/Results_PAIP_save_for_ImageJ/set3_result'
Tumor_path = join(path, 'Tumor_result')
Non_Tumor_path = join(path, 'Non_Tumor_result')

path_list = sorted(next(os.walk(Tumor_path))[2])

aim_dir = '/vast/AI_team/sukmin/Results_for_PAIP_final/Test_CAD_DL_ConvNext_large_b8_bbbaseline_DiceCE_GF_0.3_TC_0.3_r1024_223'
os.makedirs(aim_dir, exist_ok=True)

for case in path_list:
    if case[-3:] == 'png':
        T_path = join(Tumor_path, case)
        N_path = join(Non_Tumor_path, case)

        T = cv2.imread(T_path, -1)
        N = cv2.imread(N_path, -1)

        rst = np.zeros(T.shape, dtype=np.uint8)

        for x in range(T.shape[0]):
            for y in range(T.shape[1]):
                if T[x][y] == 255:
                    rst[x][y] = 1
                elif N[x][y] == 255:
                    rst[x][y] = 2

        cv2.imwrite(rst, join(aim_dir, case))

        plt.figure(figsize=(8, 6))
        plt.imshow(rst*100)
        plt.show()

#%%