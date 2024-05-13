import os
import numpy as np
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2
join = os.path.join
from PIL import Image

path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task730_Lunit_Tissue_StainNorm/labelsTs"
path_list = sorted(os.listdir(path))

aim_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task730_Lunit_Tissue_StainNorm/labelsTs_255"
os.makedirs(aim_path, exist_ok=True)



for case in path_list:
    if case[-4:]==".png":
        print(case)
        seg_p = join(path, case)
        seg = np.array(Image.open(join(path, case)))

        rst = np.ones((1024, 1024), dtype=np.uint8)

        for x in range(seg.shape[0]):
            for y in range(seg.shape[1]):
                # print(seg[x][y])
                if seg[x][y] == 1: # BGR
                    rst[x][y] = 255

        cv2.imwrite(join(aim_path,case), rst)
        print(rst.max())

        

                
    

