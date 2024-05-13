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
            # # 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
            # (h, w) = rst.shape[:2]
            # (cX, cY) = (w // 2, h // 2)
            # # 이미지를 중심으로 -90도 회전
            # M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
            # rst = cv2.warpAffine(rst, M, (w, h))


            cv2.imwrite(rst, join(args.output_path, 'case_' + img_name.split('_')[-1]))
            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {rst.shape}; costing: {t1-t0:.2f}s')

            if args.visualize_well:
                rst = rst * 50
                cv2.imwrite(rst, join(args.output_path_for_visualize, 'case_' + img_name.split('_')[-1]))

            # if args.show_overlay:
                # boundary = segmentation.find_boundaries(test_pred_mask, connectivity=1, mode='inner')
                # boundary = morphology.binary_dilation(boundary, morphology.disk(2))
                # img_data[boundary, :] = 255
                # io.imsave(join(args.output_path, 'overlay_' + img_name), img_data, check_contrast=False)



if __name__ == "__main__":
    parser = ap.ArgumentParser()
    # parser.add_argument('-t', dest='trainer', default='Liver')
    # parser.add_argument('-i', dest='input_path', default="/home/sukmin/datasets/Task106_Liver/imagesTs")
    parser.add_argument('-t', dest='trainer', default='Stroma')
    parser.add_argument('-i', dest='input_path', default="/vast/AI_team/sukmin/datasets/Task200_Stroma/imagesTs") # Task113_Carotid_side
    parser.add_argument('-o', dest='aim_path', default="Stroma_UNet_b8_b") # Liver_UNet_b16_b
    parser.add_argument('-n', dest='epoch_num', default='model_best.pth') # model_e36000.pth # model_best.pth
    parser.add_argument('-visualize_well', required=False, default=True, action="store_true", help='visualize well')
    # parser.add_argument('-show_overlay', required=False, default=True, action="store_true", help='save segmentation overlay')
    
    args = parser.parse_args()

    args.output_path = join('/vast/AI_team/sukmin/Results', args.aim_path)
    args.output_path_for_visualize = join('/vast/AI_team/sukmin/Results_visualize', args.aim_path)
    if os.path.isdir(args.output_path)== False:
        os.makedirs(args.output_path, exist_ok=True)
    info, config = call_config(args.trainer)
    os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    main(info, config, args)