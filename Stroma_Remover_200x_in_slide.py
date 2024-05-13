import os
import torch
import argparse as ap
import numpy as np
from monai.inferers import sliding_window_inference
from monai.data import *
from monai.transforms import *
from core.utils import *
from core.call_model import *
from config.call import call_config
from skimage import io, segmentation, morphology, measure, exposure
import time
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


def Stroma_Remover(input_path, output_path, model_path, gpu):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.makedirs(output_path, exist_ok=True)
    # os.makedirs(join(output_path, "images"), exist_ok=True)
    # os.makedirs(join(output_path, "norm_images"), exist_ok=True)

    # 모델 정의 (Update시 변경 예정)
    from core.model.decoders.CAD_deeplabv3_for_ConvNext import CAD_deeplabv3Plus_for_ConvNext
    model = CAD_deeplabv3Plus_for_ConvNext(
        encoder_name="tu-convnext_large",
        classes=4,
        activation='softmax2d',
        encoder_depth = 4
    )

    model = model.to("cuda")
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    roi_size = [256, 256]
    sw_batch_size = 4

    input_list = sorted(os.listdir(input_path))

    with torch.no_grad():
        for diag in input_list[2:]:
            diag_list = sorted(os.listdir(join(input_path, diag)))
            for case in diag_list:
                output_path_dir = join(output_path, diag, case)
                os.makedirs(output_path_dir, exist_ok=True)
                img_names = sorted(os.listdir(join(input_path, diag, case, 't')))
                for img_name in img_names:
                    if img_name != 'Thumbs.db':
                        img_data = io.imread(join(input_path, diag, case, 't', img_name))

                        # normalize image data
                        pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
                        for i in range(3):
                            img_channel_i = img_data[:,:,i]
                            if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                                pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
                        
                        t0 = time.time()
                        test_npy01 = pre_img_data/np.max(pre_img_data)
                        test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
                        test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)
                        test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
                        test_pred_npy = test_pred_out[0].cpu().numpy()
                        rst = np.argmax(test_pred_npy, axis=0)
                        t1 = time.time()
                        print(f'Prediction finished: {img_name}; img size = {rst.shape}; costing: {t1-t0:.2f}s')


                        # Make White Space
                        image = cv2.imread(join(input_path, diag, case, 't', img_name))
                        rst = np.repeat(rst[..., np.newaxis], 3, -1)
                        for x in range(rst.shape[0]):
                            for y in range(rst.shape[1]):
                                if rst[x][y][0] != 2:
                                    image[x][y] = [255, 255, 255]  # back-ground
                                
                        t2 = time.time()
                        print(f'Make WhiteSpace finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
                        cv2.imwrite(image, join(output_path_dir, img_name))


                        # io.read 빨간색으로 보임
                        # cv2.read 보라색으로 보임

            
if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-i', dest='input_path', default='/vast/AI_team/youngjin/data/stomach_new/10/256_rf4_2201_2302_train')                        # '/vast/AI_team/sukmin/datasets/Task505_Stroma_NDM_prep_200x_whole/imagesTs') 
    parser.add_argument('-o', dest='output_path', default= '/vast/AI_team/youngjin/data/stomach_new/10/256_rf4_2201_2302_train_Epi_only')                                                                                    # "/vast/AI_team/sukmin/Results_Stroma_Remover/SeeDP/Colon_patch_CacoX_200x_e40k/test/D") 
    parser.add_argument('-m', dest='model_path', default='/vast/AI_team/sukmin/Results_monai_Stroma_NDM/CacoX_50x_only_SE_overlap25_r256/model_best_2430.pth') 
    parser.add_argument('-gpu', dest='gpu_number', default="0")

    args = parser.parse_args()

    Stroma_Remover(args.input_path, args.output_path, args.model_path, args.gpu_number)