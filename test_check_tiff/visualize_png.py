#%%
from PIL import Image
from platform import python_version
import cv2
import numpy as np
import tifffile as tif

# import matplotlib
# matplotlib.use('Qt5Agg')
import os
join = os.path.join


# test a single image
# img = '/home/sukmin/datasets/Task150_NeurIPS_Cellseg/imagesTs/cell_00001.png'
# gt = '/home/sukmin/datasets/Task150_NeurIPS_Cellseg/labelsTs/cell_00001.png'
# rst = '/vast/AI_team/sukmin/Results/Neurips_Cellseg_CASwin_b8_b/cell_00001.png'
rst = '/vast/AI_team/sukmin/datasets/Test_Norm/Cellseg/Task001_Test_Norm_Cellseg_Not_norm/labelsTr/cell_00001.png'
# rst2 = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task732_Lunit_Tissue_StainNorm_Only1/labelsTs/case_007.png'

aim_dir = '/vast/AI_team/sukmin/datasets/visual_test'
os.makedirs(aim_dir, exist_ok=True)



# Let's take a look at the dataset
import cv2

# img = cv2.imread(img)
# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.bgr2rgb(img))
# plt.show()

# print(img.shape)


# gt = cv2.imread(gt) * 100
# plt.figure(figsize=(8, 6))
# plt.imshow(gt)
# plt.show()

# print(gt.shape)
# print(gt.max())
# print(gt.ndim)
# print(gt.dtype)



rst = cv2.imread(rst)
rst = rst*100
print(rst.max())
Image.fromarray(rst).save(join(aim_dir, "test.png"))

# rst2 = cv2.imread(rst2)
# rst2 = rst2*100
# Image.fromarray(rst2).save(join(aim_dir, "test2.png"))

# plt.figure(figsize=(8, 6))
# plt.imshow(rst)
# plt.show()

# print(rst.shape)
# print(rst.max())
# print(rst.ndim)
# print(rst.dtype)



# rst_png = cv2.imread(rst_png)
# rst_png = rst_png*100
# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.bgr2rgb(rst_png))
# plt.show()

# print(rst_png.shape)
# print(rst_png.max())
# print(rst_png.ndim)
# print(rst_png.dtype)
# %%
