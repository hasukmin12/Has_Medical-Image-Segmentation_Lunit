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
# https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py



path = '/vast/AI_team/sukmin/datasets/PAIP_valid_for_me/labels/tr_c001.png'
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


# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
image = ndi.distance_transform_edt(image)
image = image.astype(np.uint32)

# Find the local maxima in the image
local_maxi = peak_local_max(image, indices=False, footprint=np.ones((3, 3)), labels=image)
# Perform the watershed segmentation
markers = ndi.label(local_maxi)[0]
labels = watershed(-image, markers, mask=image)

labels = np.float32(labels)
# lab_image = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)

plt.figure(figsize=(8, 6))
plt.imshow(labels)
plt.show()




image2 = ndi.distance_transform_edt(image2)
image2 = image2.astype(np.uint32)

# Find the local maxima in the image
local_maxi2 = peak_local_max(image2, indices=False, footprint=np.ones((3, 3)), labels=image)
# Perform the watershed segmentation
markers2 = ndi.label(local_maxi2)
markers2 = markers2[0]
labels2 = watershed(-image2, markers2, mask=image)

labels2 = np.float32(labels2)
# lab_image = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)

plt.figure(figsize=(8, 6))
plt.imshow(labels2)
plt.show()





# markers2[~image2] = -1
# labels_rw2 = random_walker(image2, markers2)




# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.bgr2rgb(labels))
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.bgr2rgb(labels2))
# plt.show()

# print(labels_rw.max())
# print(labels_rw2.max())


c_1 = labels.max()
c_2 = labels2.max()
print("Tumor : ", c_1)
print("Non-Tumor : ", c_2)


tc = (c_1 / (c_1 + c_2)) * 100
print("TC_value : ", round(tc))





















# %%
