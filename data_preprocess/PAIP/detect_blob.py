#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = '/vast/AI_team/sukmin/datasets/PAIP_valid_for_me/labels/tr_p004.png'

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


# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 0
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

detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
keypoints = detector.detect(image)

params.minArea = 0
detector = cv2.SimpleBlobDetector_create(params)
keypoints2 = detector.detect(image2)


print("Tumor : ", len(keypoints))
print("Non-Tumor : ", len(keypoints2))

c_1 = len(keypoints)
c_2 = len(keypoints2)
tc = (c_1 / (c_1 + c_2)) * 100
print("TC_value : ", round(tc))

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
