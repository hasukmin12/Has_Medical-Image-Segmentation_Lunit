import os
import numpy as np
import cv2
import cv2
join = os.path.join


path = "/vast/AI_team/sukmin/datasets/2._PAIP2023_Traning_masks/tumor/tr_p009_tumor.png"

img = cv2.imread(path, -1)
print(img.max())



        

