import os
import numpy as np
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
join = os.path.join
import shutil

img_path = '/vast/AI_team/sukmin/datasets/Task701_Lunit_Cell_only/labelsTs'
img_list = sorted(next(os.walk(img_path))[2])

aim_dir = '/vast/AI_team/sukmin/datasets/Task701_Lunit_Cell_only/For_inference/labelsTs'
os.makedirs(aim_dir, exist_ok=True)

for case in img_list:
    if case[-4:]==".png":
        print(case)
        input = join(img_path, case)
        output = join(aim_dir, "case_{0:03d}.png".format(img_list.index(case)))
        shutil.copyfile(input, output)






        # img_data = cv2.imread(img, -1)
        # io.imsave(join(aim_dir, case), test_npy01.astype(np.uint8), check_contrast=False)
    