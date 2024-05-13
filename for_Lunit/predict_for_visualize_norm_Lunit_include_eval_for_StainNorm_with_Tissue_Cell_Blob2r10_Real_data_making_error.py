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
import json
from ultralytics import YOLO
import torch.nn.functional as F


PIL.Image.MAX_IMAGE_PIXELS = 933120000 
join = os.path.join
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# def normalize_channel(img, lower=1, upper=99):
#     non_zero_vals = img[np.nonzero(img)]
#     percentiles = np.percentile(non_zero_vals, [lower, upper])
#     if percentiles[1] - percentiles[0] > 0.001:
#         img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
#     else:
#         img_norm = img
#     return img_norm.astype(np.uint8)


def read_json(fpath: Path) -> dict:
    """This function reads a json file

    Parameters
    ----------
    fpath: Path
        path to the json file

    Returns
    -------
    dict:
        loaded data 
    """
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data


def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # # unmix hematoxylin and eosin
    # H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    # H[H>255] = 254
    # H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    # E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    # E[E>255] = 254
    # E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    return Inorm
























def main(info, config, args):

    os.makedirs(args.output_path, exist_ok=True)
    img_names = sorted(os.listdir(args.input_path))
    t_img_names = sorted(os.listdir(args.tissue_input_path))
    # c_img_names = sorted(os.listdir(args.cell_input_path))

    if "Thumbs.db" in img_names:
        img_names.remove("Thumbs.db")
    if "Thumbs.db" in t_img_names:
        t_img_names.remove("Thumbs.db")
    # if "Thumbs.db" in c_img_names:
    #     c_img_names.remove("Thumbs.db")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Upload Models
    model = call_model(info, config).to("cuda")
    model.load_state_dict(torch.load(os.path.join(info["LOGDIR"],args.epoch_num))["model_state_dict"])
    model.eval()
    
    t_model = smp.DeepLabV3Plus(
            encoder_name="se_resnext101_32x4d", 
            encoder_weights="imagenet",
            in_channels=3,
            classes=2,
        ).to(device)
    t_model.load_state_dict(torch.load(args.tissue_model_path)["model_state_dict"])
    t_model.eval()

    c_model = smp.DeepLabV3Plus(
            encoder_name="se_resnext101_32x4d", 
            encoder_weights="imagenet",
            in_channels=3,
            classes=2,
        ).to(device)
    c_model.load_state_dict(torch.load(args.cell_model_path)["model_state_dict"])
    c_model.eval()

    sw_batch_size = 1

    with torch.no_grad():
        for img_name in img_names:
            if img_name != 'Thumbs.db':
                img_data = np.array(Image.open(join(args.input_path, img_name)))
                print(img_name)
                # tissue_data = np.array(Image.open(join(args.tissue_input_path, img_name)))

                t0 = time.time()

                # StainNormalization
                test_npy01 = normalizeStaining(img_data)

                # # tissue는 이미 stain된 이미지를 불러옴
                # Stained_tissue = tissue_data/255
                Stained_cell = test_npy01/255

                # test = torch.from_numpy(np.expand_dims(np.expand_dims(tissue_data, 0), 0))
                # test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
                # t_tensor = torch.from_numpy(np.expand_dims(Stained_tissue, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
                c_tensor = torch.from_numpy(np.expand_dims(Stained_cell, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
                
                # # tissue prediction
                # tissue_pred_out = t_model(t_tensor)
                # tissue_pred_out = torch.nn.functional.softmax(tissue_pred_out, dim=1)
                # tissue_pred_out = tissue_pred_out[0].cpu().numpy()
                # tissue_pred_out = np.argmax(tissue_pred_out, axis=0)

                # # Crop as MetaData and Upsampling Image
                # meta = read_json(args.metadata_path)
                # print(int(img_names.index(img_name)+81))
                # cx = meta['sample_pairs']["{0:03d}".format(int(img_names.index(img_name)+81))]['patch_x_offset']
                # cy = meta['sample_pairs']["{0:03d}".format(int(img_names.index(img_name)+81))]['patch_y_offset']

                # # 256*256 박스 생성 중심점(cx, cy)
                # x_i = int(1024*cx-128)
                # x_o = int(1024*cx+128)
                # y_i = int(1024*cy-128)
                # y_o = int(1024*cy+128)
                # crop_tissue_patch = tissue_pred_out[y_i:y_o, x_i:x_o]
                # # crop_tissue_patch = crop_tissue_patch.repeat(4, axis=0).repeat(4, axis=1) # nearest neighbor tiling  
                # tissue_seg = torch.from_numpy(np.expand_dims(crop_tissue_patch, 0)).type(torch.FloatTensor).to(device)
                # tissue_seg = tissue_seg.unsqueeze(0) # [1, 1, 256, 256]
                # tissue_seg = tissue_seg

                # # Upsampling
                # tissue_seg= F.interpolate(
                #     tissue_seg, size=(1024,1024), mode="bilinear", align_corners=True
                # ).detach()
        
                # # save for check
                # vis_tis_seg = tissue_seg[0][0].cpu().numpy().astype(np.uint8)
                # check_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task750_Lunit_Cell_StainNorm_Blob_Cell_to_r10/Test_in_training_crop/error_prediction_list_split/tissue"
                # os.makedirs(check_path, exist_ok=True)
                # Image.fromarray(vis_tis_seg*255).save(join(check_path, img_name))
                # print(tissue_seg.max())



                # Cell prediction
                cell_pred_out = c_model(c_tensor)
                cell_pred_out = torch.nn.functional.softmax(cell_pred_out, dim=1)
                cell_pred_out = cell_pred_out[0].cpu().numpy()
                cell_pred_out = np.argmax(cell_pred_out, axis=0)

                cell_seg = cell_pred_out
                vis_cell = cell_pred_out.astype(np.uint8)
                # print(cell_seg.max())
                print()
                # vis_cell = np.reshape(cell_pred_out, (1024,1024,3)).astype(np.uint8)
                
                # cell_seg = cv2.cvtColor(cell_seg, cv2.COLOR_RGB2GRAY)

                # save for check
                check_path_c = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task750_Lunit_Cell_StainNorm_Blob_Cell_to_r10/Test_in_training_crop/error_prediction_list_split/cell"
                os.makedirs(check_path_c, exist_ok=True)
                Image.fromarray(vis_cell*255).save(join(check_path_c, img_name))
                # Image.fromarray(test_npy01).save(join(check_path, "cell_img_" + img_name))

                # Make it to tensor
                cell_seg = torch.from_numpy(np.expand_dims(cell_seg, 0)).type(torch.FloatTensor).to(device)
                cell_seg = cell_seg.unsqueeze(0)

                # # concat three images
                # test_tensor = test_tensor/255
                # tissue_seg = tissue_seg
                # cell_seg = cell_seg
                # test_tensor = torch.cat((test_tensor, tissue_seg, cell_seg), dim=1)


                # # inference Final model
                # test_pred_out = model(test_tensor)
                # # test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)[0]

                # test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
                # # test_pred_npy = test_pred_out[0,1].cpu().numpy()
                # test_pred_npy = test_pred_out[0].cpu().numpy()
                # rst = np.argmax(test_pred_npy, axis=0)

                # # rst = cv2.rotate(rst, cv2.ROTATE_90_CLOCKWISE)
                # # rst = cv2.flip(rst, 1)

                # cv2.imwrite(join(args.output_path, 'case_' + img_name.split('_')[-1]), rst)
                # # Image.fromarray(rst).save(join(args.output_path, 'case_' + img_name.split('_')[-1]))
                
                # t1 = time.time()
                # print(f'Prediction finished: {img_name}; img size = {rst.shape}; costing: {t1-t0:.2f}s')


    #             if args.make_WhiteSpace:

    #                 # image = cv2.imread(join(args.input_path, img_name))
    #                 image = np.array(Image.open(join(args.input_path, img_name)))

    #                 rst = np.repeat(rst[..., np.newaxis], 3, -1)
    #                 for x in range(rst.shape[0]):
    #                     for y in range(rst.shape[1]):
    #                         if rst[x][y][0] == 2:
    #                             image[x][y] = [255, 255, 255]  # back-ground

                            
    #                 t2 = time.time()
    #                 print(f'Make WhiteSpace finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
    #                 cv2.imwrite(join(args.output_path_for_WhiteSpace_rst, img_name), image)
    #                 # Image.fromarray(image).save(join(args.output_path_for_WhiteSpace_rst, img_name))




    #             if args.show_overlay:

    #                 # image = cv2.imread(join(args.input_path, img_name))
    #                 image = np.array(Image.open(join(args.input_path, img_name)))

    #                 from copy import deepcopy
    #                 rst2 = deepcopy(rst)

    #                 rst = np.repeat(rst[..., np.newaxis], 3, -1)
    #                 for x in range(rst.shape[0]):
    #                     for y in range(rst.shape[1]):
    #                         if rst[x][y][0] == 0:
    #                             rst[x][y] = [255, 255, 255]  # back-ground

    #                         elif rst[x][y][0] == 1: # BGR
    #                             rst[x][y] = [149, 184, 135]   # BGR  - BC (Green)
    #                         elif rst[x][y][0] == 2:
    #                             rst[x][y] = [69, 48, 237]  # TC (Red)
                

    #                 alpha = 0.6
    #                 rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
    #                 # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)

    #                 t2 = time.time()
    #                 print(f'Colored finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
    #                 cv2.imwrite(join(args.output_path_for_visualize, 'vis_' + img_name), rst)
    #                 # Image.fromarray(rst).save(join(args.output_path_for_visualize, 'seg_' + img_name))





    #             if args.for_sudo_labeling:

    #                 for x in range(rst2.shape[0]):
    #                     for y in range(rst2.shape[1]):
    #                         if rst2[x][y] == 1:      # BGR 
    #                             rst2[x][y] = 50  # stroma 
    #                         elif rst2[x][y] == 2:
    #                             rst2[x][y] = 120  # epithelium 
    #                         elif rst2[x][y] == 3:
    #                             rst2[x][y] = 100  # others 
    #                         elif rst2[x][y] == 4:
    #                             rst2[x][y] = 170  # D
    #                         elif rst2[x][y] == 5:
    #                             rst2[x][y] = 200  # M
    #                         elif rst2[x][y] == 6:
    #                             rst2[x][y] = 250  # NET

    #                 t3 = time.time()
    #                 print(f'for sudo_labeling Colored finished: {img_name}; img size = {rst2.shape}; costing: {t3-t2:.2f}s')
    #                 cv2.imwrite(join(args.output_path_for_sudo_label, 'seg_' + img_name), rst2)
    #                 # Image.fromarray(rst2).save(join(args.output_path_for_sudo_label, 'seg_' + img_name))

    # # 비교하려는 GT 폴더 결과
    # folder_with_gt = args.folder_to_inference

    # # inference 된 결과
    # folder_with_pred = args.output_path
    # labels = (0, 1, 2) # test 하고 싶은 라벨 입력

    # evaluate_folder(folder_with_gt, folder_with_pred, labels)     
                        


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-t', dest='trainer', default='Lunit_3_concat_Tissue_Cell_r10_with_StainNorm_Blob_Cell_No_MONAI_BlobCell_to_r10_with_error') # 'Stroma_NDM_100x_overlap')                                                             # 2022_stomach_400x
    parser.add_argument('-i', dest='input_path', default='/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task701_Lunit_Cell_only/For_inference_Split/imagesTs') # Task701_Lunit_Cell_only/For_inference/imagesTs  
    parser.add_argument('-o', dest='aim_path', default="Lunit_3_concat_Tissue_Cell_r10_with_StainNorm_Blob_Cell_No_MONAI_BlobCell_to_r10_Aug_Focal_errorP_Real_data_b25k_t") # Test_200x_CacoX_b32_e45k  
    parser.add_argument('-n', dest='epoch_num', default='model_best_e06000_dice0.7209060290362767.pth')  # model_e06000.pth   # model_best_e09900.pth  # class_model_best_e10001.pth
    parser.add_argument('-f_gt', dest='folder_to_inference', default='/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task723_Lunit_Cell_StainNorm_rad10/labelsTs')
    
    parser.add_argument('-i_t', dest='tissue_input_path', default='/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task730_Lunit_Tissue_StainNorm/imagesTs')

    parser.add_argument('-m_t', dest='tissue_model_path', default='/vast/AI_team/sukmin/Results_monai_Lunit_Challenge_Tissue_only/Lunit_Tissue_only_StainNorm_DL_se_ResNext_b8_Only1_v2/model_best_e05300_dice0.7937519115142727.pth') # best 모델로 선정
    parser.add_argument('-m_c', dest='cell_model_path', default='/vast/AI_team/sukmin/Results_monai_Lunit_Challenge/Lunit_Cell_only_StainNorm_DL_se_ResNext_b8_JustCell_BlobCell_Split/model_best_e07600_dice0.7121158149927108.pth')

    parser.add_argument('-meta', dest='metadata_path', default='/vast/AI_team/sukmin/datasets/Lunit_Challenge/ocelot2023_v0.1.1/metadata.json')


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
