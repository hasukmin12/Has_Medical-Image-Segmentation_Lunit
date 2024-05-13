import os
import numpy as np
import cv2
import cv2
from skimage import io
join = os.path.join


path = "/vast/AI_team/sukmin/datasets/Task301_PAIP_norm/imagesTr"
path_list = sorted(next(os.walk(path))[2])

gt_path = "/vast/AI_team/sukmin/datasets/Task301_PAIP_norm/labelsTr"
gt_list = sorted(next(os.walk(gt_path))[2])

aim_dir = '/vast/AI_team/sukmin/datasets/Task_PAIP_2023_norm_512_split/images'
os.makedirs(aim_dir, exist_ok=True)

aim_seg_dir = '/vast/AI_team/sukmin/datasets/Task_PAIP_2023_norm_512_split/labels'
os.makedirs(aim_dir, exist_ok=True)



for case in path_list:
    if case[-3:]=='png':
        print(case)
        img_path = join(path, case)
        gt_path2 = join(gt_path, case)

        img = cv2.imread(img_path)
        seg = cv2.imread(gt_path2, -1)

        img_a1 = img[:512, :512]
        seg_a1 = seg[:512, :512].astype(np.uint8)

        img_a2 = img[512:, :512]
        seg_a2 = seg[512:, :512].astype(np.uint8)

        img_a3 = img[:512, 512:]
        seg_a3 = seg[:512, 512:].astype(np.uint8)

        img_a4 = img[512:, 512:]
        seg_a4 = seg[512:, 512:].astype(np.uint8)

        # print(seg_a4.max())
        # if 1 in seg_a4:
        #     print("yes")


        cv2.imwrite(img_a1, join(aim_dir, case[:4] + '{0:03d}_0.png'.format(path_list.index(case))))
        cv2.imwrite(img_a2, join(aim_dir, case[:4] + '{0:03d}_1.png'.format(path_list.index(case))))
        cv2.imwrite(img_a3, join(aim_dir, case[:4] + '{0:03d}_2.png'.format(path_list.index(case))))
        cv2.imwrite(img_a4, join(aim_dir, case[:4] + '{0:03d}_3.png'.format(path_list.index(case))))

        cv2.imwrite(seg_a1, join(aim_seg_dir, case[:4] + '{0:03d}_0.png'.format(path_list.index(case))))
        cv2.imwrite(seg_a2, join(aim_seg_dir, case[:4] + '{0:03d}_1.png'.format(path_list.index(case))))
        cv2.imwrite(seg_a3, join(aim_seg_dir, case[:4] + '{0:03d}_2.png'.format(path_list.index(case))))
        cv2.imwrite(seg_a4, join(aim_seg_dir, case[:4] + '{0:03d}_3.png'.format(path_list.index(case))))



        # io.imsave(join(aim_seg_dir, case[:4] + '{0:03d}_3.png'.format(path_list.index(case))), seg_a1.astype(np.uint8), check_contrast=False)
        # io.imsave(join(aim_seg_dir, case[:4] + '{0:03d}_3.png'.format(path_list.index(case))), seg_a2.astype(np.uint8), check_contrast=False)
        # io.imsave(join(aim_seg_dir, case[:4] + '{0:03d}_3.png'.format(path_list.index(case))), seg_a3.astype(np.uint8), check_contrast=False)
        # io.imsave(join(aim_seg_dir, case[:4] + '{0:03d}_3.png'.format(path_list.index(case))), seg_a4.astype(np.uint8), check_contrast=False)

