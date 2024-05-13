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

from core.monai_sliding_window_inference_for_DTC import sliding_window_inference_for_tanh
from monai.data import *
from monai.transforms import *
#from monai.handlers.utils import from_engine

from core.utils import *
from core.call_model import *
from config.call import call_config
from skimage import io, segmentation, morphology, measure, exposure
join = os.path.join
import time
import tifffile as tif


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)



def main(info, config, args):

    os.makedirs(args.output_path, exist_ok=True)
    img_names = sorted(os.listdir(join(args.input_path)))

    model = call_model(info, config).to("cuda")
    model.load_state_dict(torch.load(os.path.join(info["LOGDIR"],args.epoch_num))["model_state_dict"])
    model.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4

    with torch.no_grad():
        for img_name in img_names:
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(args.input_path, img_name))
            else:
                img_data = io.imread(join(args.input_path, img_name))
            
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
            test_pred_out = sliding_window_inference_for_tanh(test_tensor, roi_size, sw_batch_size, model)[0]
            test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
            test_pred_npy = test_pred_out[0,1].cpu().numpy()
            # convert probability map to binary mask and apply morphological postprocessing
            test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(test_pred_npy>0.5),16))
            tif.imwrite(join(args.output_path, img_name.split('.')[0]+'_label.tiff'), test_pred_mask, compression='zlib')
            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {pre_img_data.shape}; costing: {t1-t0:.2f}s')
            
            if args.show_overlay:
                boundary = segmentation.find_boundaries(test_pred_mask, connectivity=1, mode='inner')
                boundary = morphology.binary_dilation(boundary, morphology.disk(2))
                img_data[boundary, :] = 255
                io.imsave(join(args.output_path, 'overlay_' + img_name), img_data, check_contrast=False)
            
      


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-t', dest='trainer', default='Neurips_Cellseg_tanh')
    parser.add_argument('-i', dest='input_path', default="/home/sukmin/datasets/Turing_test_set/images")
    parser.add_argument('-o', dest='output_path', default="/vast/AI_team/sukmin/Results_for_challenge/CAswin_deep_dense_b8_tanh")
    parser.add_argument('-n', dest='epoch_num', default='model_best.pth') # model_e50000.pth # model_best.pth
    parser.add_argument('-input_size', default=512, type=int, help='segmentation classes')
    parser.add_argument('-show_overlay', required=False, default=False, action="store_true", help='save segmentation overlay')
    args = parser.parse_args()
    if os.path.isdir(args.output_path)== False:
        os.makedirs(args.output_path, exist_ok=True)
    info, config = call_config(args.trainer)
    os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    main(info, config, args)

