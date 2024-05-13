import os
import numpy as np
import cv2
import cv2
join = os.path.join


dir_path = '/vast/AI_team/sukmin/datasets/PAIP_valid_for_me'
path = join(dir_path, 'labels')
path_list = sorted(next(os.walk(path))[2])

aim_path_1 = join(dir_path, 'Tumor_orgin')
aim_path_2 = join(dir_path, 'Non-Tumor_orgin')

os.makedirs(aim_path_1, exist_ok=True)
os.makedirs(aim_path_2, exist_ok=True)

for case in path_list:
    print(case)
    img_path = join(path, case)
    image = cv2.imread(img_path, -1)
    image2 = np.zeros(image.shape, dtype=np.uint8)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] == 1:
                image[x][y] = 255
            elif image[x][y] == 2:
                image[x][y] = 0
                image2[x][y] = 255  

    
    cv2.imwrite(image, join(aim_path_1, case))
    cv2.imwrite(image2, join(aim_path_2, case))








        

