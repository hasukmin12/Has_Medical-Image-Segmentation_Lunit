#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed, random_walker
from skimage.feature import peak_local_max


# follow belowed link
# https://github.com/Connor323/Cancer-Cell-Tracking/blob/master/Code/watershed.py



path = '/vast/AI_team/sukmin/datasets/PAIP_valid_for_me/labels/tr_p010.png'
# orgin_img_path = '/vast/AI_team/sukmin/datasets/PAIP_valid_for_me/images/tr_p004.png'

# orgin_image = cv2.imread(orgin_img_path)

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




# blob 분리하는 코드

plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(image))
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(image2))
plt.show()


dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 0)
result_dist_transform = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
ret, sure_fg = cv2.threshold(dist_transform, 4, 255, cv2.THRESH_BINARY)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(image, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.bgr2rgb(dist_transform))
# plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(result_dist_transform))
plt.show()


image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
markers = cv2.watershed(image, markers)


plt.figure(figsize=(8, 6))
plt.imshow(markers)
plt.show()










dist_transform2 = cv2.distanceTransform(image2, cv2.DIST_L2, 0)
result_dist_transform2 = cv2.normalize(dist_transform2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
ret2, sure_fg2 = cv2.threshold(dist_transform2, 0.1*dist_transform2.max(), 255, cv2.THRESH_BINARY)
sure_fg2 = np.uint8(sure_fg2)
unknown2 = cv2.subtract(image2, sure_fg2)

# Marker labelling
ret2, markers2 = cv2.connectedComponents(sure_fg2)

# Add one to all labels so that sure background is not 0, but 1
markers2 = markers2+1

# Now, mark the region of unknown with zero
markers2[unknown2==255] = 0

# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.bgr2rgb(dist_transform2))
# plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(result_dist_transform2))
plt.show()


image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
markers2 = cv2.watershed(image2, markers2)


plt.figure(figsize=(8, 6))
plt.imshow(markers2)
plt.show()




c_1 = markers.max()
c_2 = markers2.max()
print("Tumor : ", c_1)
print("Non-Tumor : ", c_2)


tc = (c_1 / (c_1 + c_2)) * 100
print("TC_value : ", round(tc))



# %%
