import os
import numpy as np
import cv2
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
join = os.path.join

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

# img = '/vast/AI_team/sukmin/datasets/1. PAIP2023_Training/tr_p003.png'
# tumor_gt = '/vast/AI_team/sukmin/datasets/2. PAIP2023_Traning masks/tumor/tr_c003_tumor.png'

img_path = '/vast/AI_team/sukmin/datasets/1. PAIP2023_Training'
img_list = sorted(next(os.walk(img_path))[2])

non_tumor_path = '/vast/AI_team/sukmin/datasets/2. PAIP2023_Traning masks/non_tumor'
non_tumor_list = sorted(next(os.walk(non_tumor_path))[2])

tumor_path = '/vast/AI_team/sukmin/datasets/2. PAIP2023_Traning masks/tumor'
tumor_list = sorted(next(os.walk(tumor_path))[2])

aim_dir = '/vast/AI_team/sukmin/datasets/Task_PAIP_2023_norm2/images'
gt_dir = "/vast/AI_team/sukmin/datasets/Task_PAIP_2023_norm2/labels"
os.makedirs(aim_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)


for case in img_list:
    if case[-4:]==".png":
        print(case)
        img = join(img_path, case)
        img_data = cv2.imread(img, -1)

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
        
        test_npy01 = pre_img_data/np.max(pre_img_data)
        # test_npy01 = pre_img_data
        
        io.imsave(join(aim_dir, case), test_npy01.astype(np.uint8), check_contrast=False)
        




        non_tumor_p = join(non_tumor_path, case[:7]+'_nontumor.png')
        tumor_p = join(tumor_path, case[:7]+'_tumor.png')

        non_tumor = cv2.imread(non_tumor_p, -1)
        tumor = cv2.imread(tumor_p, -1)

        # print(non_tumor.max())
        # print(tumor.max())

        seg = np.zeros(tumor.shape, dtype=np.uint8)
        for x in range(seg.shape[0]):
            for y in range(seg.shape[1]):
                if tumor[x][y] != 0:
                    seg[x][y] = 1
                elif non_tumor[x][y] != 0:
                    seg[x][y] = 2


        # print(seg.max())
        if seg.max() == 0:
            print("case 0 in it : ",case)
        cv2.imwrite(seg, join(gt_dir, case))



