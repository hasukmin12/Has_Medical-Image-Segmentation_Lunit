import glob
import os, tempfile
import random
from requests import post
import torch
import wandb
import argparse as ap
import numpy as np
import nibabel as nib
import yaml
from tqdm import tqdm
from typing import Tuple

from monai.inferers import sliding_window_inference
from monai.data import *
from monai.transforms import *
#from monai.handlers.utils import from_engine

from core.utils import *
from core.call_model import *
from config.call import call_config
from skimage import io, segmentation, morphology, measure, exposure
import time
import tifffile as tif
import cv2
import cv2

join = os.path.join


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


# def apply_mask(image, mask, color, alpha=0.5):
#     """Apply the given mask to the image.
#     """
#     for c in range(3):
#         image[:, :, c] = np.where(mask == 1,
#                                   image[:, :, c] *
#                                   (1 - alpha) + alpha * color[c] * 255,
#                                   image[:, :, c])
#     return image

def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5, 
    resize: Tuple[int, int] = None # (1024, 1024)
) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.
    
    Returns:
        image_combined: The combined image.
        
    """
    color = np.asarray(color).reshape(3, 1, 1)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    
    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)
    
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)[0]
    image_combined = image_combined.transpose(1, 2, 0)
    
    return image_combined





def main(info, config, args):

    os.makedirs(args.output_path, exist_ok=True)
    img_names = sorted(os.listdir(join(args.input_path)))

    model = call_model(info, config).to("cuda")
    model.load_state_dict(torch.load(os.path.join(info["LOGDIR"],args.epoch_num))["model_state_dict"])
    model.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4

    with torch.no_grad():
        import csv
        f = open(join(args.output_path, 'tc_value.csv'), 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow(["ID", "TC"])
        for img_name in img_names:
            if img_name[-4:] =='.png':
                img_data = cv2.imread(join(args.input_path, img_name), -1)

                # normalize image data
                if len(img_data.shape) == 2:
                    img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
                elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                    img_data = img_data[:,:, :3]
                else:
                    pass
                pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)

                for i in range(3):
                    img_channel_i = img_data[:,:,i]
                    if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                        pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
                
                t0 = time.time()

                test_npy01 = pre_img_data/np.max(pre_img_data)

                test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
                if info["Deep_Supervision"]:
                    test_pred_out = sliding_window_inference(
                        test_tensor, config["INPUT_SHAPE"], sw_batch_size, model)[0]
                else:
                    test_pred_out = sliding_window_inference(
                        test_tensor, config["INPUT_SHAPE"], sw_batch_size, model)
                

                # GF는 빼줘야함
                test_pred_out = test_pred_out[0]


                test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
                # test_pred_npy = test_pred_out[0,1].cpu().numpy()
                test_pred_npy = test_pred_out[0].cpu().numpy()
                rst = np.argmax(test_pred_npy, axis=0)

                import copy
                rst2 = copy.deepcopy(rst)

                vol_1 = np.zeros(rst2.shape)
                vol_2 = rst2
                for x in range(rst2.shape[0]):
                    for y in range(rst2.shape[1]):
                        if rst2[x][y] == 1:
                            vol_1[x][y] = 1
                            vol_2[x][y] = 0

                # object small holes와 object 추가하는 코드 작성
                vol_1 = morphology.remove_small_objects(morphology.remove_small_holes(vol_1>0.5),16)
                vol_2 = morphology.remove_small_objects(morphology.remove_small_holes(vol_2>0.5),16)

                for x in range(rst2.shape[0]):
                    for y in range(rst2.shape[1]):
                        if vol_1[x][y] == True:
                            rst[x][y] = 1
                        elif vol_2[x][y] == True:
                            rst[x][y] = 2
                        else:
                            rst[x][y] = 0
                

                # rst = cv2.rotate(rst, cv2.ROTATE_90_CLOCKWISE)
                # rst = cv2.flip(rst, 1)

                cv2.imwrite(rst, join(args.output_path, 'te_' + img_name.split('_')[-1]))
                t1 = time.time()
                print(f'Prediction finished: {img_name}; img size = {rst.shape}; costing: {t1-t0:.2f}s')



                if args.show_overlay:

                    image = cv2.imread(join(args.input_path, img_name))

                    rst = np.repeat(rst[..., np.newaxis], 3, -1)
                    for x in range(rst.shape[0]):
                        for y in range(rst.shape[1]):
                            if rst[x][y][0] == 0:
                                rst[x][y] = [255, 255, 255]  # back-ground

                            elif rst[x][y][0] == 1:
                                rst[x][y] = [133, 133, 218]  # BGR  - stroma (핑크)
                            elif rst[x][y][0] == 2:
                                rst[x][y] = [149, 184, 135]  # epithelium (연두)


                    alpha = 0.6
                    rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
                    # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)

                    t2 = time.time()
                    print(f'Colored finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
                    cv2.imwrite(rst, join(args.output_path_for_visualize, 'seg_' + img_name))
                



               
                rst2_1 = measure.label(vol_1)            
                rst2_2 = measure.label(vol_2)

                # 후처리 작업 시작

                # 1. Blob Detector 가동
                
                # Set up the detector with default parameters.
                params = cv2.SimpleBlobDetector_Params()

                # Filter by Area.
                params.filterByArea = True
                params.minArea = 120
                params.maxArea = 8000
                
                # Filter by Circularity
                params.filterByCircularity = True
                params.minCircularity = 0.1
                # params.maxCircularity = 0.9
                
                # Filter by Convexity
                params.filterByConvexity = True
                params.minConvexity = 0.1
                
                # Filter by Inertia
                params.filterByInertia = True
                params.filterByColor = False
                params.minInertiaRatio = 0.01

                # others
                params.minDistBetweenBlobs = 1

                detector = cv2.SimpleBlobDetector_create(params)
                
                # Detect blobs.
                vol_1_1 = np.zeros(rst2_1.shape, dtype='uint8')
                vol_2_2 = np.zeros(rst2_2.shape, dtype='uint8')

                for x in range(rst2_1.shape[0]):
                    for y in range(rst2_1.shape[1]):
                        if rst2_1[x][y] != 0:
                            vol_1_1[x][y] = 255  

                for x in range(rst2_2.shape[0]):
                    for y in range(rst2_2.shape[1]):
                        if rst2_2[x][y] != 0:
                            vol_2_2[x][y] = 255  




                # # 2. 붙어있는 Blob 떼어놓기
                # vol_1_1
                # vol_2_2






                c_1 = len(detector.detect(vol_1_1))

                # Non-Tumor의 minArea param을 바꿔줘야한다.
                params.minArea = 50
                detector = cv2.SimpleBlobDetector_create(params)            
                c_2 = len(detector.detect(vol_2_2))

                tc = (c_1 / (c_1 + c_2)) * 100
                # 소수점 반올림
                tc = round(tc)

                wr.writerow([img_name[:-4], tc])

                print(f'seperate finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
                cv2.imwrite(rst2_1, join(args.output_path_for_seperate_results, 'c1_' + img_name))
                cv2.imwrite(rst2_2, join(args.output_path_for_seperate_results, 'c2_' + img_name))


        f.close()

            

# '/vast/AI_team/sukmin/datasets/1. PAIP2023_Training'

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-t', dest='trainer', default='PAIP_2023_norm_GF')
    parser.add_argument('-i', dest='input_path', default='/vast/AI_team/sukmin/datasets/PAIP2023_Test') # 1.PAIP2023_Validation 
    parser.add_argument('-o', dest='aim_path', default="Test_CAD_DL_se_resnet152_b16_bbaseline_b_param_50") # Val_DeepLabV3+_res101_b32_e14.5k_b
    parser.add_argument('-n', dest='epoch_num', default='model_best_2500.pth') # model_e02000.pth # model_best_2500.pth 
    # parser.add_argument('-visualize_well', required=False, default=True, action="store_true", help='visualize well')
    parser.add_argument('-show_overlay', required=False, default=True, action="store_true", help='save segmentation overlay')
    # parser.add_argument('-count_tc', required=False, default=True, action="store_true", help='count tc rate for PAIP challenge')

    args = parser.parse_args()

    args.output_path = join('/vast/AI_team/sukmin/Results_PAIP_2023', args.aim_path)
    args.output_path_for_visualize = join('/vast/AI_team/sukmin/Results_visualize_PAIP_2023', args.aim_path)
    args.output_path_for_seperate_results = join('/vast/AI_team/sukmin/Results_visualize_seperate_PAIP_2023', args.aim_path)

    os.makedirs(args.output_path, exist_ok=True)
    if args.show_overlay:
        os.makedirs(args.output_path_for_visualize, exist_ok=True)

    info, config = call_config(args.trainer)
    os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    main(info, config, args)