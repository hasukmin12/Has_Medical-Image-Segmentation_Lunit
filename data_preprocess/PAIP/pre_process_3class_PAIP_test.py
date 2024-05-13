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


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]  
    percentiles = np.percentile(non_zero_vals, [lower, upper])  
                                                                
                                                                
    if percentiles[1] - percentiles[0] > 0.001:  # 
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)




def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]  # non-zero인 값만 뽑아낸다
    percentiles = np.percentile(non_zero_vals, [lower, upper])  # 이미지의 intensity 중 분위수[lower, upper]를 나타내준다.
                                                                # lower = 1, 백분위 1%의 intensity -> 크기 순서로 나열한 자료를 100등분했을때, 1%가 그 값보다 작거나 같게되는 값 = (거의 최솟값이라는 뜻)
                                                                # upper = 99, 백분위 99%의 intensity -> 크기 순서로 나열한 자료를 100등분했을때, 99%가 그 값보다 작거나 같게되는 값 = (거의 최댓값이라는 뜻)
    if percentiles[1] - percentiles[0] > 0.001:  # 
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior
    
def main():
    parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='/vast/AI_team/sukmin/datasets/PAIP2023_Test', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='/vast/AI_team/sukmin/datasets/PAIP2023_Test_prep', type=str, help='preprocessing data path')    
    args = parser.parse_args()
    
    source_path = args.input_path
    target_path = args.output_path
    
    img_path = source_path
    img_names = sorted(os.listdir(img_path))

    pre_img_path = target_path
    os.makedirs(pre_img_path, exist_ok=True)
    
    for img_name in img_names:
        if img_name[-4:] == '.png':
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(img_path, img_name))
            else:
                img_data = io.imread(join(img_path, img_name))
            
            # normalize image data
            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:,:, :3]
            else:
                pass
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                    pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
            
            io.imsave(join(target_path, img_name), pre_img_data.astype(np.uint8), check_contrast=False)
    
if __name__ == "__main__":
    main()























