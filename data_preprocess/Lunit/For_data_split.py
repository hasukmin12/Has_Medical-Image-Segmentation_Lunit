import os
import numpy as np
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
join = os.path.join
import shutil


img_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task750_Lunit_Cell_StainNorm_Blob_Cell_to_r10/Test_in_training_crop/error_prediction_list_split/cell'
img_list = sorted(next(os.walk(img_path))[2])

aim_dir = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task761_Lunit_Cell_StainNorm_BlobCell_3_concat_Split/error_prediction/imagesTr'
os.makedirs(aim_dir, exist_ok=True)

train_patients = img_list[:103] + img_list[122:191] + img_list[208:270] + img_list[290:327] + img_list[337:365] + img_list[373:394]
val_patients = img_list[103:122] + img_list[191:208] + img_list[270:290] + img_list[327:337] + img_list[365:373] + img_list[394:400]


# /vast/AI_team/sukmin/datasets/Lunit_Challenge/Task750_Lunit_Cell_StainNorm_Blob_Cell_to_r10/Test_in_training_crop/error_prediction_list_split/cell

for case in train_patients:
    if case[-4:]==".png":
        print(case)
        input = join(img_path, case)
        output = join(aim_dir, "case_{0:03d}.png".format(img_list.index(case)))
        shutil.copyfile(input, output)























# img_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task730_Lunit_Tissue_StainNorm/images'
# img_list = sorted(next(os.walk(img_path))[2])

# aim_dir = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task730_Lunit_Tissue_StainNorm/splits/imagesTr'
# os.makedirs(aim_dir, exist_ok=True)

# train_patients = img_list[:103] + img_list[122:191] + img_list[208:270] + img_list[290:327] + img_list[337:365] + img_list[373:394]
# val_patients = img_list[103:122] + img_list[191:208] + img_list[270:290] + img_list[327:337] + img_list[365:373] + img_list[394:400]


# # /vast/AI_team/sukmin/datasets/Lunit_Challenge/Task750_Lunit_Cell_StainNorm_Blob_Cell_to_r10/Test_in_training_crop/error_prediction_list_split/cell

# for case in train_patients:
#     if case[-4:]==".png":
#         print(case)
#         input = join(img_path, case)
#         output = join(aim_dir, "case_{0:03d}.png".format(img_list.index(case)))
#         shutil.copyfile(input, output)






        # img_data = cv2.imread(img, -1)
        # io.imsave(join(aim_dir, case), test_npy01.astype(np.uint8), check_contrast=False)
    