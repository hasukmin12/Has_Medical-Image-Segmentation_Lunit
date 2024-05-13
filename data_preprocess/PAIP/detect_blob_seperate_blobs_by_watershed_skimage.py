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
distance = ndi.distance_transform_edt(image)
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=image)

labels = np.float32(labels)
# labels = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)


# markers[~image] = -1
# labels_rw = random_walker(image, markers)


distance2 = ndi.distance_transform_edt(image2)
coords2 = peak_local_max(distance2, footprint=np.ones((3, 3)), labels=image2)
mask2 = np.zeros(distance2.shape, dtype=bool)
mask2[tuple(coords2.T)] = True
markers2, _ = ndi.label(mask2)
labels2 = watershed(-distance2, markers2, mask=image2)

labels2 = np.float32(labels2)
# labels2 = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)

# markers2[~image2] = -1
# labels_rw2 = random_walker(image2, markers2)




plt.figure(figsize=(8, 6))
plt.imshow(labels)
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(labels2)
# plt.show()

# print(labels_rw.max())
# print(labels_rw2.max())


c_1 = labels.max()
c_2 = labels2.max()
print("Tumor : ", c_1)
print("Non-Tumor : ", c_2)


tc = (c_1 / (c_1 + c_2)) * 100
print("TC_value : ", round(tc))







# fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(image, cmap=plt.cm.gray)
# ax[0].set_title('Overlapping objects')
# ax[1].imshow(-distance, cmap=plt.cm.gray)
# ax[1].set_title('Distances')
# ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
# ax[2].set_title('Separated objects')

# for a in ax:
#     a.set_axis_off()

# fig.tight_layout()
# plt.show()



















# # Set up the detector with default parameters.
# params = cv2.SimpleBlobDetector_Params()

# # Filter by Area.
# params.filterByArea = True
# params.minArea = 0
# params.maxArea = 8000
 
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
# # params.maxCircularity = 0.9
 
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.1
 
# # Filter by Inertia
# params.filterByInertia = True
# params.filterByColor = False
# params.minInertiaRatio = 0.01

# # others
# params.minDistBetweenBlobs = 1

# detector = cv2.SimpleBlobDetector_create(params)
 
# # Detect blobs.
# keypoints = detector.detect(image)

# params.minArea = 0
# detector = cv2.SimpleBlobDetector_create(params)
# keypoints2 = detector.detect(image2)


# print("Tumor : ", len(keypoints))
# print("Non-Tumor : ", len(keypoints2))

# c_1 = len(keypoints)
# c_2 = len(keypoints2)
# tc = (c_1 / (c_1 + c_2)) * 100
# print("TC_value : ", round(tc))

# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # Show keypoints
# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.bgr2rgb(im_with_keypoints))
# plt.show()


# # cv2.imshow("Keypoints", im_with_keypoints)
# # cv2.waitKey(0)







# %%
