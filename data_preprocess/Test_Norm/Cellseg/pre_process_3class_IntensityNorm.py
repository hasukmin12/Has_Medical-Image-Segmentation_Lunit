#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:12:04 2022

convert instance labels to three class labels:
0: background
1: interior
2: boundary
@author: jma
"""

import os
join = os.path.join
import argparse

from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
from PIL import Image
import staintools
# from StainNorm_Fuction import *

def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
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
        
    if ODhat.size == 0 or 3:
        return img
    

    elif len(ODhat) == 1:
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
        
        # unmix hematoxylin and eosin
        H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
        H[H>255] = 254
        H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
        
        E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
        E[E>255] = 254
        E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
        
        # if saveFile is not None:
        #     Image.fromarray(Inorm).save(saveFile+'.png')
        #     Image.fromarray(H).save(saveFile+'_H.png')
        #     Image.fromarray(E).save(saveFile+'_E.png')

        return Inorm # , H, E

def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior
    


# https://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues
def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr


def main():
    parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='/vast/AI_team/sukmin/datasets/Test_Norm/Cellseg/Training', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='/vast/AI_team/sukmin/datasets/Test_Norm/Cellseg/Training/Training_prep_IntensityNorm', type=str, help='preprocessing data path')    
    args = parser.parse_args()
    
    source_path = args.input_path
    target_path = args.output_path
    
    img_path = join(source_path, 'images')
    gt_path =  join(source_path, 'labels')
    
    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split('.')[0]+'_label.tiff' for img_name in img_names]

    if "Thumbs.db" in img_names:
        img_names.remove("Thumbs.db")
    if "Thumbs_label.tiff" in gt_names:
        gt_names.remove("Thumbs_label.tiff")
    
    pre_img_path = join(target_path, 'images')
    pre_gt_path = join(target_path, 'labels')
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(pre_gt_path, exist_ok=True)
    
    for img_name, gt_name in zip(tqdm(img_names), gt_names):
        if img_name.endswith('.tif') or img_name.endswith('.tiff'):
            img_data = tif.imread(join(img_path, img_name))
        else:
            img_data = io.imread(join(img_path, img_name))
        gt_data = tif.imread(join(gt_path, gt_name))
        
        # normalize image data
        if len(img_data.shape) == 2:
            img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
        elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
            img_data = img_data[:,:, :3]
        else:
            pass
        
        pre_img_data = normalize(img_data)



        # normalizer = staintools.StainNormalizer(method='vahadane')
        # normalizer.fit(target)
        # pre_img_data = normalizer.transform(to_transform)


        # conver instance bask to three-class mask: interior, boundary
        interior_map = create_interior_map(gt_data.astype(np.int16))
        
        io.imsave(join(target_path, 'images', img_name.split('.')[0]+'.png'), pre_img_data.astype(np.uint8), check_contrast=False)
        io.imsave(join(target_path, 'labels', gt_name.split('_')[0] + '_' + gt_name.split('_')[1]+'.png'), interior_map.astype(np.uint8), check_contrast=False)
    
if __name__ == "__main__":
    main()























