#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
join = os.path.join
import argparse
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2
import cv2

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


    
def main():
    parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='/vast/AI_team/sukmin/datasets/2차annotation/200x(Downsample1)', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='/vast/AI_team/sukmin/datasets/Task_Stroma_prep_200x_only_SE_overlap25_r1024', type=str, help='preprocessing data path')    
    args = parser.parse_args()
    
    input_path = args.input_path
    aim_path = args.output_path

    pre_img_path = join(aim_path, 'images')
    pre_gt_path = join(aim_path, 'labels')
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(pre_gt_path, exist_ok=True)

    val_img_path = join(aim_path, 'val_images')
    val_gt_path = join(aim_path, 'val_labels')
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(val_gt_path, exist_ok=True)



    NDM_list = sorted(next(os.walk(input_path))[1])

    case_num = 0
    N_c = 0
    D_c = 0
    M_c = 0
    NET_c = 0

    for block in NDM_list:
        print()
        print()
        print(block)
        block_path = join(input_path, block)
        folder_list = sorted(next(os.walk(block_path))[1])

        for folder in folder_list:
            folder_path = join(block_path, folder)
            case_list = sorted(next(os.walk(folder_path))[2])

            # 여기서부터가 파일 접근
            for case in case_list:
                if case != 'Thumbs.db':

                    if case[-9:] != 'label.png':
                        img_path = join(folder_path, case)
                        label_path = join(folder_path, case[:-4] + '_label.png')

                        img_data = cv2.imread(img_path)
                        gt_data = cv2.imread(label_path)

                        # normalize image data
                        pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
                        for i in range(3):
                            img_channel_i = img_data[:,:,i]
                            if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                                pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)

                        image = pre_img_data

                        patch_size = 1024
                        patch_term = 768             # 435 # 85%            # 384 - 75%          # 256 - 50%            # 128  - 25%

                        width, height, _ = image.shape
                        img_patches = []
                        seg_patches = []

                        seg = np.zeros([width, height], dtype=np.uint8)

                        # label들을 학습하기 위해 value화 해놓기
                        for x in range(width):
                            for y in range(height):

                                # https://jimmy-ai.tistory.com/304
                                # 전체 배열이 따로 True인지 False인지 나오기 때문에 .all로 전부 일치해야만 pass 하도록 코드를 짜줘야한다.

                                if (gt_data[x][y] == [0, 255, 0]).all():  # back-ground  # BGR
                                    # seg[x][y] = 0
                                    continue

                                elif (gt_data[x][y] == [255, 0, 0]).all():  # stroma
                                    seg[x][y] = 1

                                elif (gt_data[x][y] == [0, 255, 255]).all():  # N-epithelium
                                    seg[x][y] = 2

                                elif (gt_data[x][y] == [66, 129, 245]).all():  # others
                                    seg[x][y] = 3

                                elif (gt_data[x][y] == [255, 255, 0]).all():  # D-epithelium
                                    seg[x][y] = 2

                                elif (gt_data[x][y] == [0, 0, 255]).all():  # M-epithelium
                                    seg[x][y] = 2

                                elif (gt_data[x][y] == [255, 153, 179]).all():  # NET
                                    seg[x][y] = 2
                                


                        



                        # Patch 나누기 작업 돌입!

                        # 1번째 (0,0) 부터
                        for x in range(0, width, patch_term):
                            for y in range(0, height, patch_term):
                                img_patch = image[x:x+patch_size, y:y+patch_size]
                                seg_patch = seg[x:x+patch_size, y:y+patch_size]
                                if img_patch.shape == (patch_size, patch_size, 3):
                                    img_patches.append(img_patch)
                                    seg_patches.append(seg_patch)


                        # 2번째 (끝, 끝) 부터 
                        for x in range(width, 0, -patch_term): # x축 기준 한쪽방향으로만
                            img_patch = image[x-patch_size:x, height-patch_size:height]
                            seg_patch = seg[x-patch_size:x, height-patch_size:height]
                            if img_patch.shape == (patch_size, patch_size, 3):
                                img_patches.append(img_patch)
                                seg_patches.append(seg_patch)

                        for y in range(height, 0, -patch_term): # y축 기준 한쪽 방향으로만
                            if y == height:  # x축 나열과 겹치는 첫번째 패치 제거
                                continue
                            else:
                                img_patch = image[width-patch_size:width, y-patch_size:y]
                                seg_patch = seg[width-patch_size:width, y-patch_size:y]
                            if img_patch.shape == (patch_size, patch_size, 3):
                                img_patches.append(img_patch)
                                seg_patches.append(seg_patch)


                        # 3번째 (0, 끝) 패치 하나만 따오기
                        img_patch = image[0:0+patch_size, height-patch_size:height]
                        seg_patch = seg[0:0+patch_size, height-patch_size:height]
                        if img_patch.shape == (patch_size, patch_size, 3):
                                img_patches.append(img_patch)
                                seg_patches.append(seg_patch)


                        # 4번째 (끝, 0) 패치 하나만 따오기
                        img_patch = image[width-patch_size:width, 0:0+patch_size]
                        seg_patch = seg[width-patch_size:width, 0:0+patch_size]
                        if img_patch.shape == (patch_size, patch_size, 3):
                                img_patches.append(img_patch)
                                seg_patches.append(seg_patch)


                        for i in range(len(img_patches)):
                            img = img_patches[i]
                            seg = seg_patches[i]

                            # 각 클래스마다 20%는 validation에 분포하도록 만듦
                            # wsi 기준으로 
                            # N  28/143장 -> 0 ~ 27 
                            # D  5/27장 -> 143 ~ 148
                            # M  3/14장 -> 170 ~ 173
                            # NET 4/23장 -> 187 ~ 191

                            if block == '1.N':
                                N_c += 1
                            elif block == '2.D':
                                D_c +=1
                            elif block == '3.M':
                                M_c +=1
                            elif block == '4.NET':
                                NET_c += 1

                            if 0<=case_num<=27 or  143<=case_num<=148 or 170<=case_num<=173 or 187<=case_num<=191:
                                cv2.imwrite(img.astype(np.uint8), join(aim_path, 'val_images', 'case_{0:04d}_{1:03d}.png'.format(case_num, i)))
                                cv2.imwrite(seg.astype(np.uint8), join(aim_path, 'val_labels', 'case_{0:04d}_{1:03d}.png'.format(case_num, i)))

                            else:
                                cv2.imwrite(img.astype(np.uint8), join(aim_path, 'images', 'case_{0:04d}_{1:03d}.png'.format(case_num, i)))
                                cv2.imwrite(seg.astype(np.uint8), join(aim_path, 'labels', 'case_{0:04d}_{1:03d}.png'.format(case_num, i)))

                            # io.imsave(join(aim_path, 'images', 'case_{0:04d}_{1:03d}.png'.format(case_num, i)), img.astype(np.uint8), check_contrast=False)
                            # io.imsave(join(aim_path, 'labels', 'case_{0:04d}_{1:03d}.png'.format(case_num, i)), seg.astype(np.uint8), check_contrast=False)

                        print(case_num)
                        case_num += 1
                        # if case_num == 172:
                        #     print("here")

    print()
    print()
    print("end of progress")
    print("total N : ", N_c)
    print("total D : ", D_c)
    print("total M : ", M_c)
    print("total NET : ", NET_c)
    print("total slide : ", N_c + D_c + M_c + NET_c)

if __name__ == "__main__":
    main()























