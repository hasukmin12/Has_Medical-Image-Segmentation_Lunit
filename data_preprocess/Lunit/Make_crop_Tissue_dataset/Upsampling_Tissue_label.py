import os
import numpy as np
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2
import json
from pathlib import Path
import os
from pathlib import Path
import numpy as np
from PIL import Image
import json
from typing import List
from skimage import exposure
import tifffile as tif
join = os.path.join


img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_Tissue_cutting_200_100/labelsTs"
img_list = sorted(next(os.walk(img_path))[2])

aim_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_Tissue_cutting_Upscaling_200_100/labelsTs"
os.makedirs(aim_path, exist_ok=True)

for case in img_list:
    if case[-4:]==".png":
        print(case)

        img = np.array(Image.open(join(img_path, case)))
        rst = cv2.resize(img, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)

        Image.fromarray(rst).save(join(aim_path, case))

