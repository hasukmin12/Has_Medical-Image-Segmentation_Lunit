import os
import shutil
join = os.path.join

path = '/vast/AI_team/sukmin/datasets/Lunit_Norm/images/train/tissue'
img_list = sorted(next(os.walk(path))[2])

aim_img_path = '/vast/AI_team/sukmin/datasets/Lunit_Norm/images/train/tissue_start_0'
os.makedirs(aim_img_path, exist_ok=True)

for case in img_list:
    if case != 'Thumbs.db':
        print(case)
        img_p = join(path, case)
        shutil.copy(img_p, join(aim_img_path, 'case_{0:03d}.png'.format(img_list.index(case))))