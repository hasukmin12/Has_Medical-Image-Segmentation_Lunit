#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2


# follow belowed link
# https://learning.rc.virginia.edu/notes/opencv/


path = '/vast/AI_team/sukmin/datasets/PAIP_valid_for_me/labels/tr_p004.png'
orgin_img_path = '/vast/AI_team/sukmin/datasets/PAIP_valid_for_me/images/tr_p004.png'

orgin_image = cv2.imread(orgin_img_path)

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


kernel = np.ones((3,3),np.uint8)
sure_bg = cv2.dilate(image,kernel,iterations=10)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(sure_bg))
plt.show()

dist_transform = cv2.distanceTransform(image ,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(dist_transform))
plt.show()



# sure_fg is float32, convert to uint8 and find unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(sure_fg))
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(unknown))
plt.show()






# label markers 
ret, markers = cv2.connectedComponents(sure_fg)

# add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(orgin_image, markers)

# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.bgr2rgb(markers))
# plt.show()


orgin_image[markers == -1] = [0,255,255]

plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(orgin_image))
plt.show()




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
