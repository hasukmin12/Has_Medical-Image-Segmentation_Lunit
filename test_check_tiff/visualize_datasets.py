#%%
# Check Pytorch installation
from PIL import Image
from platform import python_version
import cv2
# Check MMSegmentation installation
import mmseg
print(mmseg.__version__)

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import numpy as np


# test a single image
img = '/home/sukmin/datasets/Task020_NeurIPS_Cellseg/imagesTs/cell_00901.png'
seg = '/home/sukmin/datasets/Task020_NeurIPS_Cellseg/labelsTs/cell_00901.png'
rst = '/home/sukmin/Results/rst_NeurIPS_Cellseg_semi_MDK_convnext_ASPPHead_turing_prep/images_png_dir/cell_00005.png'

# img = '/home/sukmin/datasets/Task005_Liver/imagesTr/002.png'
# seg = '/home/sukmin/datasets/Task005_Liver/labelsTr/002.png'
# img = '/home/sukmin/datasets/Task001_TIGER/imagesTr/TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-4D96-43F4-9DD7-A61535786297_[22877, 12530, 24892, 13993].png'
# seg = '/home/sukmin/datasets/Task001_TIGER/labelsTr/TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-4D96-43F4-9DD7-A61535786297_[22877, 12530, 24892, 13993].png'

# Let's take a look at the dataset
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(img)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(img))
plt.show()

print(img.shape)
print(img.ndim)
print(img.dtype)

seg = cv2.imread(seg)
seg = seg*100
plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(seg))
plt.show()

print(seg.shape)
print(seg.max())
print(seg.ndim)
print(seg.dtype)



rst = cv2.imread(rst)
rst = rst*100
plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(rst))
plt.show()

print(rst.shape)
print(rst.max())
print(rst.ndim)
print(rst.dtype)

# %%
