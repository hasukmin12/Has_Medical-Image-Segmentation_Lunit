#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, segmentation, morphology, measure, exposure

path = '/vast/AI_team/sukmin/datasets/PAIP_valid_for_me/labels/tr_p001.png'

image = cv2.imread(path, -1) 
image2 = np.zeros(image.shape, dtype=np.uint8)

print(image.max())
for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        if image[x][y] == 1:
            image[x][y] = 255
        elif image[x][y] == 2:
            image[x][y] = 0
            image2[x][y] = 255  


keypoints = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(image>0.5),16))
keypoints2 = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(image2>0.5),16))



print("Tumor : ", keypoints.max())
print("Non-Tumor : ", keypoints2.max())

c_1 = keypoints.max()
c_2 = keypoints2.max()
tc = (c_1 / (c_1 + c_2)) * 100
print("TC_value : ", round(tc))





# %%
