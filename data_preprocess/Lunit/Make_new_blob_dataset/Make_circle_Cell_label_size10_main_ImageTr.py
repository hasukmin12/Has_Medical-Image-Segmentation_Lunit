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
import csv

img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task723_Lunit_Cell_StainNorm_rad10/imagesTs"
img_list = sorted(next(os.walk(img_path))[2])

anno_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_Norm/annotations/train/cell'
anno_list = sorted(next(os.walk(anno_path))[2])

aim_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTs_circle_150'
os.makedirs(aim_path, exist_ok=True)

if "Thumbs.db" in img_list:
    img_list.remove("Thumbs.db")

for case in anno_list:
    if case != 'Thumbs.db':
        print(case)
        anno = open(join(anno_path, case), 'r', encoding='utf-8')
        pt = csv.reader(anno)

        # print(img_list[anno_list.index(case)-80])
        # img = cv2.imread(join(img_path, img_list[anno_list.index(case)-80]))
        seg = np.zeros((1024, 1024), dtype=np.uint8)

        for line in pt:
            # print(line)
            x = int(line[0])
            y = int(line[1])
            cls = int(line[2])

            if cls == 1:
                cv2.circle(seg, (x,y), 10, 150, -1)
            elif cls == 2:
                cv2.circle(seg, (x,y), 10, 150, -1)

        # print(seg.max())

        # cv2.imwrite(img, join(aim_path, 'case_{0:03d}.png'.format(anno_list.index(case))))
        Image.fromarray(seg).save(join(aim_path, 'case_{0:03d}.png'.format(anno_list.index(case))))




# # import shutil
#     img_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_Not_norm/images/train/cell'
#     img_list = sorted(next(os.walk(img_path))[2])
#     img_list.remove("Thumbs.db")

# # aim_img_path = '/vast/AI_team/sukmin/datasets/Lunit_Norm/images/train/cell_start_0_5'
# # os.makedirs(aim_img_path, exist_ok=True)

# # for case in img_list:
# #     if case != 'Thumbs.db':
# #         print(case)
# #         img_p = join(img_path, case)
# #         shutil.copy(img_p, join(aim_img_path, 'cell_{0:03d}.png'.format(img_list.index(case))))





#     # cv2.imwrite(seg.astype(np.uint8), join(aim_path, 'labels', 'case_{0:04d}_{1:03d}.png'.format(case_num, i)))


#     # overlap rst
#     image = cv2.imread(join(img_path, img_list[anno_list.index(case)]))
#     rst = np.zeros((1024, 1024, 3), dtype=np.uint8)

#     for x in range(seg.shape[0]):
#         for y in range(seg.shape[1]):
#             # print(seg[x][y])
#             if seg[x][y] == 1: # BGR
#                 rst[x][y] = [149, 184, 135]  # BGR  - Background Cell (연두)
#             elif seg[x][y] == 2:
#                 rst[x][y] = [69, 48, 237]  # Tumor Cell (빨강)
            
#     # print(rst.max())
#     alpha = 0.4
#     rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
#     # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)
#     cv2.imwrite(rst, join(output_path_for_visualize, 'case_{0:03d}.png'.format(anno_list.index(case))))


#     anno.close()

        
    