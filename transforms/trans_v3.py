from monai.transforms import *
import numpy as np
import math

# for semi-supervised learning

def call_transforms_for_semi_easy_hard(config):

    hard_train_transforms = [ # Hard augmentation
            LoadImaged(
                keys=["image", "label"], dtype=np.uint8
            ),  # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
            AsChannelFirstd(
                keys=["image"], channel_dim=-1, allow_missing_keys=True
            ),  # image: (3, H, W)
            ScaleIntensityd(
                keys=["image"], allow_missing_keys=True
            ),  # Do not scale label
            SpatialPadd(keys=["image", "label"], spatial_size=config["INPUT_SHAPE"]),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=config["INPUT_SHAPE"], random_size=False
            ),
            RandAxisFlipd(keys=["image", "label"], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["image"], prob=0.5, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["image"], prob=0.5, num_control_points=3),
            RandZoomd(
                keys=["image", "label"],
                prob=0.5,
                min_zoom=0.5,
                max_zoom=2.0,
                mode=["area", "nearest"],
            ),
            EnsureTyped(keys=["image", "label"]),

            # Added Hard augmentation
            RandRicianNoised(keys=["image"], prob=0.5),
            RandCoarseShuffled(keys=["image"], prob=0.5, holes=200, spatial_size=20),
            Rand2DElasticd(keys=['image', 'label'], prob=0.5, spacing=(20,20),magnitude_range=(1,2), 
                            padding_mode='zeros', mode=['bilinear', 'nearest'])
            
        ]




    val_transforms = [
            LoadImaged(keys=["image", "label"], dtype=np.uint8),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=["image"], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["image"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["image", "label"]),
        ]

    easy_unlabel_transforms = [
            LoadImaged(keys=["image"], dtype=np.uint8),
            AsChannelFirstd(keys=["image"], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["image"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            SpatialCropd(
                keys=["image"], roi_size=config["INPUT_SHAPE"], roi_center=(512,512)
            ),
            EnsureTyped(keys=["image"]),
        ]

    hard_unlabel_transforms =[
            LoadImaged(keys=["image"], dtype=np.uint8),
            AsChannelFirstd(keys=["image"], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["image"], allow_missing_keys=True),
            SpatialCropd(
                keys=["image"], roi_size=config["INPUT_SHAPE"], roi_center=(512,512)
            ),

            # # intensity transform
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["image"], prob=0.5, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["image"], prob=0.5, num_control_points=3),
        
            # Added Hard augmentation
            RandRicianNoised(keys=["image"], prob=0.5),
            EnsureTyped(keys=["image"]),
            # RandCoarseShuffled(keys=["image"], prob=0.5, holes=200, spatial_size=20),
            # Rand2DElasticd(keys=['image'], prob=0.5, spacing=(20,20),magnitude_range=(1,2), 
            #                 padding_mode='zeros', mode=['bilinear'])
            
        ]



    if config["FAST"]:
        train_transforms += [ToDeviced(keys=["image", "label"], device="cuda:0")]
        val_transforms += [ToDeviced(keys=["image", "label"], device="cuda:0")]

    return Compose(hard_train_transforms), Compose(val_transforms), Compose(easy_unlabel_transforms), Compose(hard_unlabel_transforms)