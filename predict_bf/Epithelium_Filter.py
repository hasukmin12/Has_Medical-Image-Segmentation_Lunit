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



def Epithelium_Filter(input_path, output_path, model_path, gpu):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(join(output_path, "images"), exist_ok=True)
    os.makedirs(join(output_path, "norm_images"), exist_ok=True)

    img_names = sorted(os.listdir(join(input_path)))

    # 모델 정의 (Update시 변경 예정)
    from monai.networks.nets import UNet
    model = UNet(  # ResUNet 사용
        spatial_dims=2,
        in_channels=3,
        out_channels=4,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )

    model = model.to("cuda")
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    roi_size = [512, 512]
    sw_batch_size = 4

    with torch.no_grad():
        for img_name in img_names:
            img_data = io.imread(join(args.input_path, img_name))

            # normalize image data
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                    pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
            

            t0 = time.time()
            test_npy01 = pre_img_data/np.max(pre_img_data)
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
            test_pred_out = sliding_window_inference(
            test_tensor, roi_size, sw_batch_size, model)
            test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
            test_pred_npy = test_pred_out[0].cpu().numpy()
            rst = np.argmax(test_pred_npy, axis=0)
            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {rst.shape}; costing: {t1-t0:.2f}s')



            # Make White Space
            image = cv2.imread(join(input_path, img_name))

            rst = np.repeat(rst[..., np.newaxis], 3, -1)
            for x in range(rst.shape[0]):
                for y in range(rst.shape[1]):
                    if rst[x][y][0] != 2:
                        image[x][y] = [255, 255, 255]  # back-ground
                    
            t2 = time.time()
            print(f'Make WhiteSpace finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
            cv2.imwrite(image, join(join(output_path, "images"), img_name))




            # Overwrite Up to Normalize image
            image2 = pre_img_data # pre_img_data

            for x in range(rst.shape[0]):
                for y in range(rst.shape[1]):
                    if rst[x][y][0] != 2:
                        image2[x][y] = [255, 255, 255]  # back-ground
                    
            t3 = time.time()
            print(f'Make WhiteSpace on Normalize image finished: {img_name}; img size = {rst.shape}; costing: {t3-t2:.2f}s')
            cv2.imwrite(image2, join(join(output_path, "norm_images"), img_name))

                

            



if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-i', dest='input_path', default='/vast/AI_team/sukmin/Datasets_for_inference/2022_stomach_400x/images') 
    parser.add_argument('-o', dest='output_path', default="/vast/AI_team/sukmin/Results_WhiteSpace_data/WhiteSpace_400x_ResUNet_b32_norm_b") 
    parser.add_argument('-m', dest='model_path', default='/vast/AI_team/sukmin/Results_monai_Stroma_norm/ResUNet_b32_norm_prep8/model_best.pth') 
    parser.add_argument('-gpu', dest='gpu_number', default="0")

    args = parser.parse_args()

    Epithelium_Filter(args.input_path, args.output_path, args.model_path, args.gpu_number)