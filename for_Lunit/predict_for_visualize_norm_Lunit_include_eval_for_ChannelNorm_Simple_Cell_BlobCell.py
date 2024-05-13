#%%
import os
from requests import post
import torch
import wandb
import argparse as ap
import numpy as np
# import nibabel as nib
from tqdm import tqdm
from typing import Tuple
from PIL import Image

from monai.inferers import sliding_window_inference
from monai.data import *
from monai.transforms import *
#from monai.handlers.utils import from_engine

from core.utils import *
from core.call_model import *
from config.call import call_config
# from skimage import io, segmentation, morphology, measure, exposure
import time
import tifffile as tif
# import cv2
import cv2
import PIL
import matplotlib.pyplot as plt

from evaluation.evaluator import evaluate_folder
from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch

from core.datasets_for_lunit import call_dataloader_Lunit
from core.datasets_for_lunit_utils import *
from skimage import io, segmentation, morphology, measure, exposure


PIL.Image.MAX_IMAGE_PIXELS = 933120000 
join = os.path.join
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


# def read_json(fpath: Path) -> dict:
#     """This function reads a json file

#     Parameters
#     ----------
#     fpath: Path
#         path to the json file

#     Returns
#     -------
#     dict:
#         loaded data 
#     """
#     with open(fpath, 'r') as f:
#         data = json.load(f)
#     return data
















def main(info, config, args):

    os.makedirs(args.output_path, exist_ok=True)
    img_names = sorted(os.listdir(args.input_path))

    if "Thumbs.db" in img_names:
        img_names.remove("Thumbs.db")

    model = call_model(info, config).to("cuda")
    model.load_state_dict(torch.load(os.path.join(info["LOGDIR"],args.epoch_num))["model_state_dict"])
    model.eval()


    # t_model = CAD_deeplabv3Plus_for_ConvNext(
    #         encoder_name="tu-convnext_large",
    #         classes=config["CHANNEL_OUT"],
    #         activation='softmax2d',
    #         encoder_depth = 4,                
    #     )
    # t_model.load_state_dict(torch.load(args.tissue_model_path)["model_state_dict"])
    # t_model.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # roi_size = (args.input_size, args.input_size)
    sw_batch_size = 1

    # test_path = '/vast/AI_team/sukmin/Lunit_cell_test'
    # bf_norm_img_path = join(test_path, "bf_StainNorm_PIL_all")
    # af_norm_img_path = join(test_path, "af_StainNorm_PIL_all")
    # os.makedirs(bf_norm_img_path, exist_ok=True)
    # os.makedirs(af_norm_img_path, exist_ok=True)


    with torch.no_grad():
        for img_name in img_names:
            if img_name != 'Thumbs.db':
                # img_data = cv2.imread(join(args.input_path, img_name))
                img_data = np.array(Image.open(join(args.input_path, img_name)))

                t0 = time.time()

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

                test_npy01 = pre_img_data/np.max(pre_img_data)
                test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
                
                test_pred_out = model(test_tensor)
                # test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)[0]

                test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
                test_pred_npy = test_pred_out[0].cpu().numpy()
                rst = np.argmax(test_pred_npy, axis=0)

                cv2.imwrite(join(args.output_path, 'case_' + img_name.split('_')[-1]), rst)
                # Image.fromarray(rst).save(join(args.output_path, 'case_' + img_name.split('_')[-1]))
                
                t1 = time.time()
                print(f'Prediction finished: {img_name}; img size = {rst.shape}; costing: {t1-t0:.2f}s')


                if args.make_WhiteSpace:

                    # image = cv2.imread(join(args.input_path, img_name))
                    image = np.array(Image.open(join(args.input_path, img_name)))

                    rst = np.repeat(rst[..., np.newaxis], 3, -1)
                    for x in range(rst.shape[0]):
                        for y in range(rst.shape[1]):
                            if rst[x][y][0] == 2:
                                image[x][y] = [255, 255, 255]  # back-ground

                            
                    t2 = time.time()
                    print(f'Make WhiteSpace finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
                    cv2.imwrite(join(args.output_path_for_WhiteSpace_rst, img_name), image)
                    # Image.fromarray(image).save(join(args.output_path_for_WhiteSpace_rst, img_name))




                if args.show_overlay:

                    # image = cv2.imread(join(args.input_path, img_name))
                    image = np.array(Image.open(join(args.input_path, img_name)))

                    from copy import deepcopy
                    rst2 = deepcopy(rst)

                    rst = np.repeat(rst[..., np.newaxis], 3, -1)
                    for x in range(rst.shape[0]):
                        for y in range(rst.shape[1]):
                            if rst[x][y][0] == 0:
                                rst[x][y] = [255, 255, 255]  # back-ground

                            elif rst[x][y][0] == 1: # BGR
                                rst[x][y] = [149, 184, 135]   # BGR  - BC (Green)
                            elif rst[x][y][0] == 2:
                                rst[x][y] = [69, 48, 237]  # TC (Red)
                

                    alpha = 0.6
                    rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
                    # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)

                    t2 = time.time()
                    print(f'Colored finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
                    cv2.imwrite(join(args.output_path_for_visualize, 'vis_' + img_name), rst)
                    # Image.fromarray(rst).save(join(args.output_path_for_visualize, 'seg_' + img_name))





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
                    cv2.imwrite(join(args.output_path_for_sudo_label, 'seg_' + img_name), rst2)
                    # Image.fromarray(rst2).save(join(args.output_path_for_sudo_label, 'seg_' + img_name))

    # 비교하려는 GT 폴더 결과
    folder_with_gt = args.folder_to_inference

    # inference 된 결과
    folder_with_pred = args.output_path
    labels = (0, 1) # test 하고 싶은 라벨 입력

    evaluate_folder(folder_with_gt, folder_with_pred, labels)     
                        


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-t', dest='trainer', default='Lunit_Cell_only_StainNorm_r10_with_MONAI_Just_Cell_BlobCell_Split') # 'Stroma_NDM_100x_overlap')                                                             # 2022_stomach_400x
    parser.add_argument('-i', dest='input_path', default='/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task701_Lunit_Cell_only/For_inference/imagesTs') # Task701_Lunit_Cell_only/For_inference/imagesTs  
    parser.add_argument('-o', dest='aim_path', default="Lunit_JustCell_ChannelNorm_DL_ResNext_b8_BlobCell_MONAI_e0.6k") # Test_200x_CacoX_b32_e45k  
    parser.add_argument('-n', dest='epoch_num', default='model_best_e01200_dice0.6404347145532261.pth') # model_e06000.pth   # model_best_e09900.pth  # class_model_best_e10001.pth
    parser.add_argument('-f_gt', dest='folder_to_inference', default='/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task747_Lunit_Cell_StainNorm_BlobCell_JustCell/labelsTs')
    
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
# %%
