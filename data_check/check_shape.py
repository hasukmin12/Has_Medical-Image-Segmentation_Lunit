import os
import numpy as np
import cv2
import cv2
join = os.path.join


path = "/vast/AI_team/sukmin/datasets/Task301_PAIP_norm/labelsTr/tr_c001.png"

path2 = "/vast/AI_team/sukmin/datasets/Task302_PAIP_norm_512_split/labelsTr/tr_c001_0.png"




img = cv2.imread(path)
print(img.shape)

img2 = cv2.imread(path2)
print(img2.shape)


        

