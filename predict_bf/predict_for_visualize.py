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


# def normalize_channel(img, lower=1, upper=99):
#     non_zero_vals = img[np.nonzero(img)]
#     percentiles = np.percentile(non_zero_vals, [lower, upper])
#     if percentiles[1] - percentiles[0] > 0.001:
#         img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
#     else:
#         img_norm = img
#     return img_norm.astype(np.uint8)


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
    # img_names = sorted(os.listdir(join(args.input_path)))
    test_images = sorted(glob.glob(os.path.join(args.input_path, "*.png")))
    test_data = [{"image": image} for image in test_images]

    model = call_model(info, config).to("cuda")
    model.load_state_dict(torch.load(os.path.join(info["LOGDIR"],args.epoch_num))["model_state_dict"])
    model.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4


    test_transforms = Compose([
            LoadImaged(keys=["image"], dtype=np.uint8),
            AsChannelFirstd(keys=["image"], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["image"], allow_missing_keys=True),
            EnsureTyped(keys=["image"]),
            # RandRotate90d(keys=["image"], prob=1, spatial_axes=[0, 1]),
        ])

    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)



    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data['image'].to(device)
            img_name = test_data['image_meta_dict']['filename_or_obj'][0].split("/")[-1]
            sw_batch_size = 4
            t0 = time.time()
            if info["Deep_Supervision"]:
                test_pred_out = sliding_window_inference(
                    test_inputs, config["INPUT_SHAPE"], sw_batch_size, model)[0]
            else:
                test_pred_out = sliding_window_inference(
                    test_inputs, config["INPUT_SHAPE"], sw_batch_size, model)

            test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
            test_pred_npy = test_pred_out[0].cpu().numpy()
            rst = np.argmax(test_pred_npy, axis=0)

            rst = cv2.rotate(rst, cv2.ROTATE_90_CLOCKWISE)
            rst = cv2.flip(rst, 1)

            cv2.imwrite(rst, join(args.output_path, 'case_' + img_name.split('_')[-1]))
            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {rst.shape}; costing: {t1-t0:.2f}s')


            if args.make_WhiteSpace:

                image = cv2.imread(join(args.input_path, img_name))

                rst = np.repeat(rst[..., np.newaxis], 3, -1)
                for x in range(rst.shape[0]):
                    for y in range(rst.shape[1]):
                        if rst[x][y][0] == 2:
                            image[x][y] = [255, 255, 255]  # back-ground

                        
                t2 = time.time()
                print(f'Make WhiteSpace finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
                cv2.imwrite(image, join(args.output_path_for_WhiteSpace_rst, img_name))



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
                        elif rst[x][y][0] == 3:
                            rst[x][y] = [192, 135, 160]  # others (보라)
            

                alpha = 0.6
                rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
                # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)

                t2 = time.time()
                print(f'Colored finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
                cv2.imwrite(rst, join(args.output_path_for_visualize, 'seg_' + img_name))
                

            



if __name__ == "__main__":
    parser = ap.ArgumentParser()
    # parser.add_argument('-t', dest='trainer', default='Liver')
    # parser.add_argument('-i', dest='input_path', default="/home/sukmin/datasets/Task106_Liver/imagesTs")
    parser.add_argument('-t', dest='trainer', default='Stroma')
    parser.add_argument('-i', dest='input_path', default='/vast/AI_team/sukmin/Datasets_for_inference/2022_stomach_200x - Testset/images') # 2022_stomach_200x - Testset  # /vast/AI_team/sukmin/Datasets_for_inference/2022_stomach_400x/images
    parser.add_argument('-o', dest='aim_path', default="Test_ResUNet_b32_Dropout0.0_b") # 2022_200x_Stomach_ResUNet_b32_Dropout0.5_b
    parser.add_argument('-n', dest='epoch_num', default='model_best.pth') # model_e36000.pth # model_best.pth
    # parser.add_argument('-visualize_well', required=False, default=True, action="store_true", help='visualize well')
    parser.add_argument('-show_overlay', required=False, default=True, action="store_true", help='save segmentation overlay')
    parser.add_argument('-make_WhiteSpace', required=False, default=False, action="store_true", help='save segmentation overlay')

    args = parser.parse_args()

    args.output_path = join('/vast/AI_team/sukmin/Results', args.aim_path)
    args.output_path_for_visualize = join('/vast/AI_team/sukmin/Results_visualize', args.aim_path)
    args.output_path_for_WhiteSpace_rst = join('/vast/AI_team/sukmin/Results_WhiteSpace_data', args.aim_path)

    os.makedirs(args.output_path, exist_ok=True)
    if args.show_overlay:
        os.makedirs(args.output_path_for_visualize, exist_ok=True)
    if args.make_WhiteSpace:    
        os.makedirs(args.output_path_for_WhiteSpace_rst, exist_ok=True)

    info, config = call_config(args.trainer)
    os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    main(info, config, args)