import os
import numpy as np
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2
join = os.path.join

anno_path = '/vast/AI_team/sukmin/datasets/ocelot2023_v0.1.1/annotations/train/tissue'
anno_list = sorted(next(os.walk(anno_path))[2])

vis_dir = '/vast/AI_team/sukmin/datasets/Lunit_Not_norm/annotations/train/tissue_visualization'
os.makedirs(vis_dir, exist_ok=True)

img_path = '/vast/AI_team/sukmin/datasets/Lunit_Not_norm/images/train/tissue'
img_list = sorted(next(os.walk(img_path))[2])

aim_path = '/vast/AI_team/sukmin/datasets/Lunit_Not_norm/annotations/train/tissue_for_train_021'
os.makedirs(aim_path, exist_ok=True)

for case in anno_list:
    if case[-4:]==".png":
        print(case)
        seg_p = join(anno_path, case)
        seg = cv2.imread(seg_p, -1)
        # print(seg.max())
        # print(seg.__contains__(0))
        # print(seg.__contains__(1))
        # print(seg.__contains__(2))
        # print(seg.__contains__(255))

        rst = np.zeros((1024, 1024), dtype=np.uint8)

        for x in range(seg.shape[0]):
            for y in range(seg.shape[1]):
                # print(seg[x][y])
                if seg[x][y] == 1: 
                    rst[x][y] = 0  # Background
                elif seg[x][y] == 2:
                    rst[x][y] = 2  # Cancer Area
                elif seg[x][y] == 255:
                    rst[x][y] = 1  # 255로 찍힌거 


        cv2.imwrite(rst, join(aim_path, 'case_{0:03d}.png'.format(anno_list.index(case))))



        # rst = np.zeros((1024, 1024, 3), dtype=np.uint8)

        # for x in range(seg.shape[0]):
        #     for y in range(seg.shape[1]):
        #         # print(seg[x][y])
        #         if seg[x][y] == 1: # BGR
        #             rst[x][y] = [149, 184, 135]  # BGR  - Background Cell (초록) : 잘 안보임 ㅋㅋ 보색인듯
        #         elif seg[x][y] == 2:
        #             rst[x][y] = [69, 48, 237]  # Tumor Cell (빨강)
        #         elif seg[x][y] == 255:
        #             rst[x][y] = [232, 237, 4]  # 255로 찍힌거 (하늘)

        #         # if seg[x][y] == 1: # BGR
        #         #     rst[x][y] = [0, 242, 255]  # BGR  - Background Cell (노랑)
        #         # elif seg[x][y] == 2:
        #         #     rst[x][y] = [69, 48, 237]  # Tumor Cell (빨강)
        #         # elif seg[x][y] == 255:
        #         #     rst[x][y] = [232, 237, 4]  # 255로 찍힌거 (하늘)

        # alpha = 0.4
        # image = cv2.imread(join(img_path, img_list[anno_list.index(case)]))
        # rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
        # # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)
        
        

    

