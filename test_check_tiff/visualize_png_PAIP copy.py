#%%
from PIL import Image
from platform import python_version
import cv2
import numpy as np
import tifffile as tif


# # test a single image
img = '/vast/AI_team/sukmin/datasets/Task301_PAIP_norm/imagesTr/tr_c002.png'
tumor_gt = '/vast/AI_team/sukmin/datasets/Task301_PAIP_norm/labelsTr/tr_c002.png'
non_tumor_gt = '/vast/AI_team/sukmin/datasets/Task301_PAIP_norm/labelsTr/tr_c002.png'


# img = '/vast/AI_team/sukmin/datasets/1. PAIP2023_Training/tr_p001.png'
# tumor_gt = '/vast/AI_team/sukmin/datasets/2. PAIP2023_Traning masks/tumor/tr_c001_tumor.png'
# non_tumor_gt = '/vast/AI_team/sukmin/datasets/2. PAIP2023_Traning masks/non_tumor/tr_c001_nontumor.png'


# Let's take a look at the dataset
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(img, -1)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(img))
plt.show()

print(img.shape)


gt = cv2.imread(tumor_gt, -1)
plt.figure(figsize=(8, 6))
plt.imshow(gt)
plt.show()

print(gt.shape)
print(gt.max())
print(gt.ndim)
print(gt.dtype)



rst = cv2.imread(non_tumor_gt, -1)
plt.figure(figsize=(8, 6))
plt.imshow(rst)
plt.show()

print(rst.shape)
print(rst.max())
print(rst.ndim)
print(rst.dtype)



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
