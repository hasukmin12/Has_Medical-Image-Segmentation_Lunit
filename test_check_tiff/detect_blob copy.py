#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2

tumor_gt = '/vast/AI_team/sukmin/Results_visualize_seperate_PAIP_2023/Val_CAD_DL_ConvNext_large_b16_baseline_b/c1_val_p008.png'
non_tumor_gt = '/vast/AI_team/sukmin/Results_visualize_seperate_PAIP_2023/Val_CAD_DL_ConvNext_large_b16_baseline_b/c2_val_p008.png'


image = cv2.imread(tumor_gt, -1) 

for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        if image[x][y] != 0:
            image[x][y] = 255  


# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 120
params.maxArea = 8000
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
# params.maxCircularity = 0.9
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1
 
# Filter by Inertia
params.filterByInertia = True
params.filterByColor = False
params.minInertiaRatio = 0.01

# others
params.minDistBetweenBlobs = 1


# parameter.filterByArea = True
# parameter.filterByConvexity = True
# parameter.filterByCircularity = True
# parameter.filterByInertia = False
# parameter.filterByColor = False
# parameter.minArea = 120  # this value defines the minimum size of the blob
# parameter.maxArea = 8000  # this value defines the maximum size of the blob
# parameter.minDistBetweenBlobs = 1 # not used in binary image
# parameter.minConvexity = 0.3
# parameter.minCircularity = 0.3
detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
keypoints = detector.detect(image)
print(len(keypoints))
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(im_with_keypoints))
plt.show()


# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)







# %%
