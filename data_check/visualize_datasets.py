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
import numpy as np


# test a single image
seg1 = '/vast/AI_team/sukmin/datasets/Task500_Stroma_NDM_prep_200x/labelsTr/case_0028_000.png'
seg2 = '/vast/AI_team/sukmin/datasets/Task500_Stroma_NDM_prep_200x/labelsTs/case_0144_000.png'
seg3 = '/vast/AI_team/sukmin/datasets/Task500_Stroma_NDM_prep_200x/labelsTs/case_0171_000.png'
seg4 = '/vast/AI_team/sukmin/datasets/Task500_Stroma_NDM_prep_200x/labelsTs/case_0188_000.png'

# img = '/home/sukmin/datasets/Task005_Liver/imagesTr/002.png'
# seg = '/home/sukmin/datasets/Task005_Liver/labelsTr/002.png'
# img = '/home/sukmin/datasets/Task001_TIGER/imagesTr/TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-4D96-43F4-9DD7-A61535786297_[22877, 12530, 24892, 13993].png'
# seg = '/home/sukmin/datasets/Task001_TIGER/labelsTr/TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-4D96-43F4-9DD7-A61535786297_[22877, 12530, 24892, 13993].png'

# Let's take a look at the dataset
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(seg1)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(img))
plt.show()

print(img.shape)
print(img.max())

seg = cv2.imread(seg2)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(seg))
plt.show()

print(seg.shape)
print(seg.max())



seg = cv2.imread(seg3)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(seg))
plt.show()

print(seg.shape)
print(seg.max())



seg = cv2.imread(seg4)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(seg))
plt.show()

print(seg.shape)
print(seg.max())

# %%
