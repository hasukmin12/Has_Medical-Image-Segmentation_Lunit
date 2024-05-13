import os
import numpy as np
# import cv2
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2
import json
from pathlib import Path
import os
from pathlib import Path
import numpy as np
from PIL import Image
import json
from typing import List
from skimage import exposure
import tifffile as tif
import torch
join = os.path.join




img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/imagesTr"
img_list = sorted(next(os.walk(img_path))[2])

t_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_Tissue_cutting_Upscaling_255/labelsTr"
t_list = sorted(next(os.walk(t_path))[2])

c_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTr_circle_150"
c_list = sorted(next(os.walk(c_path))[2])



aim_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_3_Concat_image/imagesTr"
os.makedirs(aim_path, exist_ok=True)


if "Thumbs.db" in img_list:
    img_list.remove("Thumbs.db")

for case in img_list:
    if case[-4:]==".png":
        print(case)
    
        img_patch = np.array(Image.open(join(img_path, case)))
        tissue_patch = np.array(Image.open(join(t_path, case))).reshape(1024, 1024, 1)
        cell_patch = np.array(Image.open(join(c_path, case))).reshape(1024, 1024, 1)

        rst = np.concatenate((img_patch, tissue_patch, cell_patch), axis=2)
        # 256*256 -> 1024*1024

        Image.fromarray(rst).save(join(aim_path, case))


        





        

    

