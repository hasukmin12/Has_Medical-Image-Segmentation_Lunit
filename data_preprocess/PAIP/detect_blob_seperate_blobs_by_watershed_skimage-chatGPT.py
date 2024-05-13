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
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, segmentation

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


# Apply Gaussian blur to the image
blurred = filters.gaussian(image, sigma=1.0)

# Compute the image gradient using Sobel filter
gradient = filters.sobel(blurred)

# Perform morphological opening to remove small objects
kernel = morphology.disk(3)
opening = morphology.opening(gradient, kernel)

# Compute the watershed markers by finding the local minima in the image
markers = filters.rank.gradient(opening, morphology.disk(5)) < 4
markers = morphology.label(markers)

# Perform watershed segmentation on the gradient image
labels = segmentation.watershed(gradient, markers)




plt.figure(figsize=(8, 6))
plt.imshow(labels)
plt.show()




# Apply Gaussian blur to the image
blurred2 = filters.gaussian(image2, sigma=1.0)

# Compute the image gradient using Sobel filter
gradient2 = filters.sobel(blurred2)

# Perform morphological opening to remove small objects
kernel = morphology.disk(3)
opening2 = morphology.opening(gradient2, kernel)

# Compute the watershed markers by finding the local minima in the image
markers2 = filters.rank.gradient(opening2, morphology.disk(3)) < 5
markers2 = morphology.label(markers2)

# Perform watershed segmentation on the gradient image
labels2 = segmentation.watershed(gradient2, markers2)


plt.figure(figsize=(8, 6))
plt.imshow(labels2)
plt.show()



c_1 = labels.max()
c_2 = labels2.max()
print("Tumor : ", c_1)
print("Non-Tumor : ", c_2)


tc = (c_1 / (c_1 + c_2)) * 100
print("TC_value : ", round(tc))



# %%
