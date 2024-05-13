import time, os
import numpy as np
from tqdm import tqdm
from statistics import mean, median
import os
import numpy as np
from pathlib import Path
# from numpy.lib.shape_base import tile
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from multiprocessing import Process, JoinableQueue
from utils_tiling1 import save_patches, delete_light_patches_new
from skimage import io, segmentation, morphology, exposure
import cv2
join = os.path.join


def tiling(slide, slide_path, save_path,tile_size,overlap,ext):

    save_patches(slide_file=slide_path +'/'+ slide,
                output_path=save_path,
                resolution_factor=0,
                tile_size=tile_size,
                overlap=overlap,
                ext=ext,
                use_filter=True)
    
    delete_light_patches_new(_dir=save_path, ext=ext, threshold = 2048*80)




def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)



# raw_data_path = '/vast/AI_team/youngjin/data/stomach/slide/stomach_slide_original/train/'
raw_data_path = '/vast/AI_team/sukmin/datasets/0424_breast_prostate/Prostate'

save_path = '/vast/AI_team/sukmin/datasets/0424_breast_prostate_patches/Prostate_200x_2'
os.makedirs(save_path, exist_ok=True)

# Norm_aim_dir = '/vast/AI_team/sukmin/datasets/0424_breast_prostate_patches/Norm_Prostate_100x_4096'
# os.makedirs(Norm_aim_dir, exist_ok=True)

ext = 'png'
tile_size = 4096
overlap = 0
levells = []
cls = []
rls = []

# 1차 Tiling
data_path = join(raw_data_path)
list = os.listdir(data_path)
for slide in  list:
        print(slide)
        tiling(slide, data_path, save_path, tile_size, overlap, ext)



# # 2차 Channel Normalization
# print()
# print()
# print()
# print("Channel Normalization Start !!")
# case_list = sorted(next(os.walk(save_path))[2])

# for case in case_list:
#     if case[-4:]==".png":
#         print(case)
#         img = join(save_path, case)
#         img_data = cv2.imread(img, -1)

#         # normalize image data
#         if len(img_data.shape) == 2:
#             img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
#         elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
#             img_data = img_data[:,:, :3]
#         else:
#             pass
#         pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
#         for i in range(3):
#             img_channel_i = img_data[:,:,i]
#             if len(img_channel_i[np.nonzero(img_channel_i)])>0:
#                 pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
        
#         io.imsave(join(Norm_aim_dir, case), pre_img_data.astype(np.uint8), check_contrast=False)
