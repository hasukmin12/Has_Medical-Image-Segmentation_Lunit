# re-read_img.py를 실행 해줘야만 라벨을 RGB로 인지한다
# https://github.com/open-mmlab/mmsegmentation/issues/550

import shutil
import os
import json
import tifffile as tif
from skimage import io, segmentation, morphology, exposure
# from batchgenerators.utilities.file_and_folder_operations import *
join = os.path.join
import numpy as np

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

if __name__ == "__main__":
    """
    This is the code from sukmin Ha
    """

    base = "/vast/AI_team/sukmin/datasets/Task_Stroma_prep_200x"

    task_id = 400
    task_name = "Stroma_NDM_prep_200x"
    foldername = "Task%03.0d_%s" % (task_id, task_name)

    nnUNet_raw_data = '/vast/AI_team/sukmin/datasets'

    out_base = join(nnUNet_raw_data, foldername)

    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")

    os.makedirs(imagestr, exist_ok=True)
    os.makedirs(imagests, exist_ok=True)
    os.makedirs(labelstr, exist_ok=True)
    os.makedirs(labelsts, exist_ok=True)

    train_patient_names = []
    val_patient_names = []
    test_patient_names = []
    img_list = next(os.walk(join(base, 'images')))[2]
    seg_list = next(os.walk(join(base, 'labels')))[2]

    val_img_list = next(os.walk(join(base, 'val_images')))[2]
    val_seg_list = next(os.walk(join(base, 'val_labels')))[2]

  
    train_patients = img_list 
    val_patients = val_img_list # 각 클래143스마다 20%는 validation에 분포하도록 만듦
    # wsi 기준으로 
    # N  28/143장 -> 0 ~ 27 
    # D  5/27장 -> 143 ~ 148
    # M  3/14장 -> 170 ~ 173
    # NET 4/23장 -> 187 ~ 191



    for p in train_patients:
        if p != "Thumbs.db":
            img_file = join(base, 'images', p)
            seg_file = join(base, 'labels', p)

            shutil.copy(img_file, join(imagestr, p))
            shutil.copy(seg_file, join(labelstr, p))
            train_patient_names.append(p)

    
    for p in val_patients:
        if p != "Thumbs.db":
            img_file = join(base, 'val_images', p)
            seg_file = join(base, 'val_labels', p)

            shutil.copy(img_file, join(imagests, p))
            shutil.copy(seg_file, join(labelsts, p))
            val_patient_names.append(p)


    # for p in test_patients:
    #     img_file = join(base, 'images', p)
    #     seg_file = join(base, 'labels', p)

    #     shutil.copy(img_file, join(imagests, p))
    #     shutil.copy(seg_file, join(labelsts, p))
    #     test_patient_names.append(p)    




    json_dict = {}
    json_dict['name'] = "Stroma - Epithelium segmentation for NDM"
    json_dict['description'] = "Sukmin Ha"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "Seegene MF"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['labels'] = {
        "0": "background",
        "1": "stroma",
        "2": "N-epithelium",
        "3": "D-epithelium",
        "4": "M-epithelium",
        "5": "NET",
        "6": "others"
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numVal'] = len(val_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s" % i.split("/")[-1], "label": "./labelsTr/%s" % i.split("/")[-1]} for i in
                         train_patient_names]
    json_dict['valid'] = [{'image': "./imagesTs/%s" % i.split("/")[-1], "label": "./labelsTs/%s" % i.split("/")[-1]} for i in
                         val_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))



# re-read_img.py를 실행 해줘야만 라벨을 RGB로 인지한다
# https://github.com/open-mmlab/mmsegmentation/issues/550
