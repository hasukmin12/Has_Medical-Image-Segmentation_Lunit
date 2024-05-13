#%%
# Check Pytorch installation
# from PIL import Image
# from platform import python_version
# import cv2
# Check MMSegmentation installation
import mmseg
print(mmseg.__version__)

# from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
# from mmseg.core.evaluation import get_palette
# import numpy as np
import tifffile as tif


# test a single image
gt = '/home/sukmin/ocelot23algo/test/input/images/cell_patches/000.tif'
# rst = '/home/sukmin/datasets/turing_test_label/cell_00001_label.tiff'
rst = gt # '/input/images/cell_patches/000.tif'
# '/home/sukmin/Results/Turing_Unet_base/cell_00001_label.tiff'
# rst = '/home/sukmin/pre_works/NeurIPS-CellSeg/CellSeg_Test/test_demo.tif'




# Let's take a look at the dataset
import cv2
import matplotlib.pyplot as plt

# img = cv2.imread(img)
# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.bgr2rgb(img))
# plt.show()

# print(img.shape)


gt = tif.imread(gt)
# plt.figure(figsize=(8, 6))
# plt.imshow(gt)
# plt.show()

print(gt.shape)
print(gt.max())
print(gt.ndim)
print(gt.dtype)



# rst = tif.imread(rst)
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
