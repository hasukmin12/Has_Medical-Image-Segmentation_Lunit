#%%
# %matplotlib inline
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2

tumor_gt = '/vast/AI_team/sukmin/Results_visualize_seperate_PAIP_2023/Val_CAD_DL_ConvNext_large_b16_baseline_b/c1_val_p008.png'
non_tumor_gt = '/vast/AI_team/sukmin/Results_visualize_seperate_PAIP_2023/Val_CAD_DL_ConvNext_large_b16_baseline_b/c2_val_p008.png'


image = cv2.imread(tumor_gt, -1)
output = image.copy()
height, width = image.shape[:2]
maxRadius = int(1.1*(width/12)/2)
minRadius = int(0.9*(width/12)/2)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(image=image, 
                           method=cv2.HOUGH_GRADIENT, 
                           dp=1, # 1.2, 
                           minDist=10, # 2*minRadius,
                           param1=10,
                           param2=10,
                           minRadius=10,
                           maxRadius=maxRadius                           
                          )

if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circlesRound = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circlesRound:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)

    plt.imshow(output)
    print(len(circles[0]))
else:
    print ('No circles found')
# %%


# circles = cv2.HoughCircles(image=image, 
#                            method=cv2.HOUGH_GRADIENT, 
#                            dp=1, # 1.2, 
#                            minDist=minRadius, # 2*minRadius,
#                            param1=50,
#                            param2=50,
#                            minRadius=minRadius,
#                            maxRadius=maxRadius                           
#                           )