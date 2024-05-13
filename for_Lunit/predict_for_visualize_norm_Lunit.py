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
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000 



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
        for img_name in img_names:
            if img_name != 'Thumbs.db':
                # img_data = io.imread(join(args.input_path, img_name))
                img_data = cv2.imread(join(args.input_path, img_name))
                # img_data_1 = cv2.imread(join(args.input_path, img_name), -1)

                # normalize image data
                if len(img_data.shape) == 2:
                    img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
                elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                    img_data = img_data[:,:, :3]
                else:
                    pass
                pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)

                # # test for sample
                # sample = np.array([51, 102, 153], dtype=np.uint8)
                # sample_norm = normalize_channel(sample, lower=1, upper=99)
                # print(sample_norm)
                # print(min(img_data[0]))

                for i in range(3):
                    img_channel_i = img_data[:,:,i]
                    if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                        pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
                
                t0 = time.time()

                test_npy01 = pre_img_data/np.max(pre_img_data)
                # print(pre_img_data.max())
                # print(test_npy01.max())

                test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
                if info["Deep_Supervision"]:
                    test_pred_out = sliding_window_inference(
                        test_tensor, config["INPUT_SHAPE"], sw_batch_size, model)[0]
                else:
                    test_pred_out = sliding_window_inference(
                        test_tensor, config["INPUT_SHAPE"], sw_batch_size, model)
                
                # test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)[0]

                test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
                # test_pred_npy = test_pred_out[0,1].cpu().numpy()
                test_pred_npy = test_pred_out[0].cpu().numpy()
                rst = np.argmax(test_pred_npy, axis=0)

                # rst = cv2.rotate(rst, cv2.ROTATE_90_CLOCKWISE)
                # rst = cv2.flip(rst, 1)

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

                    from copy import deepcopy
                    rst2 = deepcopy(rst)

                    rst = np.repeat(rst[..., np.newaxis], 3, -1)
                    for x in range(rst.shape[0]):
                        for y in range(rst.shape[1]):
                            if rst[x][y][0] == 0:
                                rst[x][y] = [255, 255, 255]  # back-ground

                            elif rst[x][y][0] == 1: # BGR
                                rst[x][y] = [149, 184, 135]  # BGR  - stroma (연두)
                            elif rst[x][y][0] == 2:
                                rst[x][y] = [0, 242, 255]  # epithelium (노랑)
                            elif rst[x][y][0] == 3:
                                rst[x][y] = [192, 135, 160]  # others (보라)
                            elif rst[x][y][0] == 4:
                                rst[x][y] = [7, 153, 237]  # D (주황)
                            elif rst[x][y][0] == 5:
                                rst[x][y] = [69, 48, 237]  # M (빨강)
                            elif rst[x][y][0] == 6:
                                rst[x][y] = [232, 237, 4]  # NET (하늘)
                

                    alpha = 0.6
                    rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
                    # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)

                    t2 = time.time()
                    print(f'Colored finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
                    cv2.imwrite(rst, join(args.output_path_for_visualize, 'seg_' + img_name))





                if args.for_sudo_labeling:

                    for x in range(rst2.shape[0]):
                        for y in range(rst2.shape[1]):
                            if rst2[x][y] == 1:      # BGR 
                                rst2[x][y] = 50  # stroma 
                            elif rst2[x][y] == 2:
                                rst2[x][y] = 120  # epithelium 
                            elif rst2[x][y] == 3:
                                rst2[x][y] = 100  # others 
                            elif rst2[x][y] == 4:
                                rst2[x][y] = 170  # D
                            elif rst2[x][y] == 5:
                                rst2[x][y] = 200  # M
                            elif rst2[x][y] == 6:
                                rst2[x][y] = 250  # NET


                    t3 = time.time()
                    print(f'for sudo_labeling Colored finished: {img_name}; img size = {rst2.shape}; costing: {t3-t2:.2f}s')
                    cv2.imwrite(rst2, join(args.output_path_for_sudo_label, 'seg_' + img_name))
                
                        
# /vast/AI_team/sukmin/datasets/QualitativeTest_NET
# /vast/AI_team/sukmin/Datasets_for_inference/2022_stomach_400x/images

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-t', dest='trainer', default='Lunit_cell_only') # 'Stroma_NDM_100x_overlap')                                                             # 2022_stomach_400x
    parser.add_argument('-i', dest='input_path', default='/vast/AI_team/sukmin/datasets/Task701_Lunit_Cell_only/For_inference/imagesTs') # 2022_stomach_200x - Testset  
    parser.add_argument('-o', dest='aim_path', default="Lunit_Only_cell_CacoX_b8_e15k") # Test_200x_CacoX_b32_e45k  
    parser.add_argument('-n', dest='epoch_num', default='model_e15000.pth') # model_e15000.pth # model_best_6296.pth
    parser.add_argument('-show_overlay', required=False, default=True, action="store_true", help='save segmentation overlay')
    parser.add_argument('-make_WhiteSpace', required=False, default=False, action="store_true", help='stroma become whitespace and save it')
    parser.add_argument('-for_sudo_labeling', required=False, default=False, action="store_true", help='inference for labeling')

    args = parser.parse_args()

    args.output_path = join('/vast/AI_team/sukmin/Results_Lunit', args.aim_path)
    args.output_path_for_visualize = join('/vast/AI_team/sukmin/Results_visualize_Lunit', args.aim_path)
    # args.output_path_for_WhiteSpace_rst = join('/vast/AI_team/sukmin/Results_WhiteSpace_data', args.aim_path)
    # args.output_path_for_sudo_label = join('/vast/AI_team/sukmin/Results_for_sudo_label', args.aim_path)

    os.makedirs(args.output_path, exist_ok=True)
    if args.show_overlay:
        os.makedirs(args.output_path_for_visualize, exist_ok=True)
    if args.make_WhiteSpace:    
        os.makedirs(args.output_path_for_WhiteSpace_rst, exist_ok=True)

    info, config = call_config(args.trainer)
    os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    main(info, config, args)