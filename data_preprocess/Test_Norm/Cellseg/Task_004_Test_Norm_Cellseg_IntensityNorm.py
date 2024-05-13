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

    base = "/vast/AI_team/sukmin/datasets/Test_Norm/Cellseg"

    task_id = 4
    task_name = "Test_Norm_Cellseg_IntensityNorm"
    foldername = "Task%03.0d_%s" % (task_id, task_name)

    nnUNet_raw_data = '/vast/AI_team/sukmin/datasets/Test_Norm/Cellseg'

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
    train_img_path = join(base, 'Training', 'Training_prep_IntensityNorm', 'images')
    train_seg_path = join(base, 'Training', 'Training_prep_IntensityNorm', 'labels')

    train_img_list = next(os.walk(train_img_path))[2]
    train_seg_list = next(os.walk(train_seg_path))[2]

    if "Thumbs.db" in train_img_list:
        train_img_list.remove("Thumbs.db")
    if "Thumbs.db" in train_seg_list:
        train_seg_list.remove("Thumbs.db")



    val_img_path = join(base, 'Tuning', 'Tuning_prep_IntensityNorm', 'images')
    val_seg_path = join(base, 'Tuning', 'Tuning_prep_IntensityNorm', 'labels')

    val_img_list = next(os.walk(val_img_path))[2]
    val_seg_list = next(os.walk(val_seg_path))[2]

    if "Thumbs.db" in val_img_list:
        val_img_list.remove("Thumbs.db")
    if "Thumbs.db" in val_seg_list:
        val_seg_list.remove("Thumbs.db")

    

    train_patients = train_img_list
    val_patients = val_img_list


    for p in train_patients:
        if p != "Thumbs.db":
            img_file = join(train_img_path, p)
            seg_file = join(train_seg_path, p)

            shutil.copy(img_file, join(imagestr, p))
            shutil.copy(seg_file, join(labelstr, p))
            train_patient_names.append(p)

    
    for p in val_patients:
        if p != "Thumbs.db":
            img_file = join(val_img_path, p)
            seg_file = join(val_seg_path, p)

            shutil.copy(img_file, join(imagests, p))
            shutil.copy(seg_file, join(labelsts, p))
            val_patient_names.append(p)


    json_dict = {}
    json_dict['name'] = "Test Norm"
    json_dict['description'] = "Suk Min Ha"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "Seegene MF"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['labels'] = {
        "0": "background",
        "1": "Cells", # "BC (Background Cell)",
        "2": "Cell boundary"
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
