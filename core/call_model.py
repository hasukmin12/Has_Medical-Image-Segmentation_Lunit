import os, glob
import torch
from torch import optim
from monai.data import *
import segmentation_models_pytorch as smp


from core.model.decoders.CAD_deeplabv3 import CAD_deeplabv3Plus
from core.model.decoders.CAD_deeplabv3_for_ConvNext import CAD_deeplabv3Plus_for_ConvNext
from core.model.decoders.CAD_deeplabv3_for_ResNext import CAD_deeplabv3Plus_for_ResNext


from core.model.decoders.deeplabv3_GF import DeepLabV3Plus_GF
from core.model.decoders.CAD_deeplabv3_GF import CAD_DeepLabV3Plus_GF
from core.model.decoders.CAD_deeplabv3_GF_ConvNext import CAD_DeepLabV3Plus_GF_ConvNext
from core.model.decoders.CAD_deeplabv3_for_ConvNext.CacoX_for_count_cell import CacoX_for_Lunit
from core.model.Lunit_unet import UNet as Lunit_UNet
from core.model.decoders.deeplabv3_DeepSupervision import DeepLabV3Plus_DeepSupervision
from core.model.decoders.deeplabv3_tissue import DeepLabV3Plus_for_tissue
from core.model.decoders.deeplabv3_tissue_DeepSupervision import DeepLabV3Plus_for_tissue_DeepSupervision
from core.model.decoders.deeplabv3_tissue_DeepSupervision_b import DeepLabV3Plus_for_tissue_DeepSupervision_b

# ENCODER = 'se_resnext50_32x4d'
# ENCODER_WEIGHTS = 'imagenet'
# CLASSES = ['car']
# ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
# DEVICE = 'cuda'


def call_model(info, config):
    model = None



    if config["MODEL_NAME"] in ['DL_se_resnext101']:
        model = smp.DeepLabV3Plus(
            encoder_name="se_resnext101_32x4d", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )

    elif config["MODEL_NAME"] in ['DL_se_resnext101_DeepSupervision']:
        model = DeepLabV3Plus_DeepSupervision(
            encoder_name="se_resnext101_32x4d", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )


    elif config["MODEL_NAME"] in ['DL_se_resnext101_with_tissue']:
        model = DeepLabV3Plus_for_tissue(
            encoder_name="se_resnext101_32x4d", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )


    elif config["MODEL_NAME"] in ['DL_se_resnext101_with_tissue_DeepSupervision']:
        model = DeepLabV3Plus_for_tissue_DeepSupervision(
            encoder_name="se_resnext101_32x4d", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )


    elif config["MODEL_NAME"] in ['DL_se_resnext101_with_tissue_DeepSupervision_(b)']:
        model = DeepLabV3Plus_for_tissue_DeepSupervision_b(
            encoder_name="se_resnext101_32x4d", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )











    # pretrained model
    # create segmentation model with pretrained encoder
    elif config["MODEL_NAME"] in ['se_resnext50_pretrain']:
        model = smp.FPN(
            encoder_name='se_resnext50_32x4d', 
            encoder_weights="imagenet", 
            classes=config["CHANNEL_OUT"],
            activation='sigmoid',
        )

    elif config["MODEL_NAME"] in ['unet_resnet34_pretrain']:
        model = smp.Unet(
            encoder_name="resnet34", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )

    elif config["MODEL_NAME"] in ['unet_resnet101_pretrain']:
        model = smp.Unet(
            encoder_name="resnet101", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )


    elif config["MODEL_NAME"] in ['DeepLabV3Plus_resnet34_pretrain']:
        model = smp.DeepLabV3Plus(
            encoder_name="resnet34", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )

    elif config["MODEL_NAME"] in ['DeepLabV3Plus_resnet101_pretrain']:
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )



    elif config["MODEL_NAME"] in ['DL_resnext50']:
        model = smp.DeepLabV3Plus(
            encoder_name="resnext50_32x4d", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )

    elif config["MODEL_NAME"] in ['DL_resnext101']:
        model = smp.DeepLabV3Plus(
            encoder_name="resnext101_32x8d", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )


    elif config["MODEL_NAME"] in ['MEDIAR_pretrain']:
        from core.model.MEDIAR import MEDIARFormer
        model = MEDIARFormer(
            encoder_name="mit_b5",
            encoder_weights="imagenet",
            decoder_channels=(1024, 512, 256, 128, 64),
            decoder_pab_channels=256,
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )


    elif config["MODEL_NAME"] in ['ReMA_pretrain']:
        from core.model.MEDIAR import MEDIARFormer
        model = MEDIARFormer(
            encoder_name="resnext101_32x8d",
            encoder_weights="imagenet",
            decoder_channels=(1024, 512, 256, 128, 64),
            decoder_pab_channels=256,
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )


    # elif config["MODEL_NAME"] in ['mit_encoder_DL_decoder']:
    #     model = smp.DeepLabV3Plus(
    #         encoder_name="mit_b5", 
    #         encoder_weights="imagenet",
    #         in_channels=config["CHANNEL_IN"],
    #         classes=config["CHANNEL_OUT"],
    #     )

    elif config["MODEL_NAME"] in ['DL_resnest269e']:
        model = smp.DeepLabV3Plus(
            encoder_name="timm-resnest269e", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )


    elif config["MODEL_NAME"] in ['DL_regnetx_320']:
        model = smp.DeepLabV3Plus(
            encoder_name="timm-regnetx_320", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )

    elif config["MODEL_NAME"] in ['DL_regnety_320']:
        model = smp.DeepLabV3Plus(
            encoder_name="timm-regnety_320", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )

    









    # 최종 개발

    elif config["MODEL_NAME"] in ['DL_se_resnet152']:
        model = smp.DeepLabV3Plus(
            encoder_name="se_resnet152", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )

    elif config["MODEL_NAME"] in ['CAD_DL_se_resnet152']:  
        model = CAD_deeplabv3Plus(
            encoder_name="se_resnet152",
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )



    elif config["MODEL_NAME"] in ['DL_se_resnet152_GF']:  
        model = DeepLabV3Plus_GF(
            encoder_name="se_resnet152",
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],              
        )

    elif config["MODEL_NAME"] in ['CAD_DL_se_resnet152_GF']:  
        model = CAD_DeepLabV3Plus_GF(
            encoder_name="se_resnet152",
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],              
        )




    # ConvNext

    elif config["MODEL_NAME"] in ['DL_ConvNext_large']:  
        model = smp.DeepLabV3Plus(
            encoder_name="tu-convnext_large",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                 
        )

    elif config["MODEL_NAME"] in ['CAD_DL_ConvNext_large']:  
        model = CAD_deeplabv3Plus_for_ConvNext(
            encoder_name="tu-convnext_large",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                  
        )


    elif config["MODEL_NAME"] in ['CacoX']:  
        model = CAD_deeplabv3Plus_for_ConvNext(
            encoder_name="tu-convnext_large",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                  
        )






    
    elif config["MODEL_NAME"] in ['DL_CA_ResNext101']:  
        model = CAD_deeplabv3Plus_for_ResNext(
            encoder_name="resnext101_32x8d",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                  
        )


    

    elif config["MODEL_NAME"] in ['DL_CA_ResNext101_4d']:  
        model = CAD_deeplabv3Plus_for_ResNext(
            encoder_name="resnext101_32x4d",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                  
        )




    elif config["MODEL_NAME"] in ['DL_CA_se_ResNext101_4d']:  
        model = CAD_deeplabv3Plus_for_ResNext(
            encoder_name="se_resnext101_32x4d",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                  
        )






    # elif config["MODEL_NAME"] in ['CacoX_for_Lunit']:  
    #     model = CacoX_for_Lunit(
    #         cell_patch, tissue_patch, pair_id                
    #     )








    elif config["MODEL_NAME"] in ['DL_ConvNext_large_GF']:  
        model = DeepLabV3Plus_GF(
            encoder_name="tu-convnext_large",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                 
        )

    elif config["MODEL_NAME"] in ['DL_ConvNext_base_GF']:  
        model = DeepLabV3Plus_GF(
            encoder_name="tu-convnext_base",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                 
        )


    elif config["MODEL_NAME"] in ['CAD_DL_ConvNext_large_GF']:  
        model = CAD_DeepLabV3Plus_GF_ConvNext(
            encoder_name="tu-convnext_large",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                 
        )


    elif config["MODEL_NAME"] in ['CacoX_GF']:  
        model = CAD_DeepLabV3Plus_GF_ConvNext(
            encoder_name="tu-convnext_large",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                 
        )






    elif config["MODEL_NAME"] in ['CAD_DL_ConvNext_base_GF']:  
        model = CAD_DeepLabV3Plus_GF_ConvNext(
            encoder_name="tu-convnext_base",
            classes=config["CHANNEL_OUT"],
            activation='softmax2d',
            encoder_depth = 4,
            # dropout = config["DROPOUT"]
            # decoder_channels = (128, 64, 32, 16)                 
        )
    






    
    elif config["MODEL_NAME"] in ['Lunit_UNet']:  
        model = Lunit_UNet(
            n_channels=3,
            n_classes=config["CHANNEL_OUT"],
                 
        )























    


















    

    elif config["MODEL_NAME"] in ['DL_dpn107']:
        model = smp.DeepLabV3Plus(
            encoder_name="dpn107", 
            encoder_weights="imagenet+5k",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )

    elif config["MODEL_NAME"] in ['DL_dpn131']:
        model = smp.DeepLabV3Plus(
            encoder_name="dpn131", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )



    elif config["MODEL_NAME"] in ['DL_senet154']:
        model = smp.DeepLabV3Plus(
            encoder_name="senet154", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )


    elif config["MODEL_NAME"] in ['DL_eff-b8']:
        model = smp.DeepLabV3Plus(
            encoder_name="timm-efficientnet-b8", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )

    elif config["MODEL_NAME"] in ['DL_eff-l2']:
        model = smp.DeepLabV3Plus(
            encoder_name="timm-efficientnet-l2", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )

    elif config["MODEL_NAME"] in ['DL_inceptionresv2']:
        model = smp.DeepLabV3Plus(
            encoder_name="inceptionresnetv2", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )



























    # 이전에 구현해놨던 모델들

    elif config["MODEL_NAME"] in ['unetr', 'UNETR']:
        from monai.networks.nets import UNETR
        model = UNETR(
            in_channels = config["CHANNEL_IN"],
            out_channels = config["CHANNEL_OUT"],
            img_size = config["INPUT_SHAPE"],
            feature_size = config["PATCH_SIZE"],
            hidden_size = config["EMBED_DIM"],
            mlp_dim = config["MLP_DIM"],
            num_heads = config["NUM_HEADS"],
            dropout_rate = config["DROPOUT"],
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
        )
    elif config["MODEL_NAME"] in ['unetr_pretrained', 'UNETR_PreTrained']:
        from core.model.UNETR import call_pretrained_unetr
        model = call_pretrained_unetr(info, config)
    elif config["MODEL_NAME"] in ['vnet', 'VNET', 'VNet', 'Vnet']:
        from monai.networks.nets import VNet
        model = VNet(
            spatial_dims=2,
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
        )





    elif config["MODEL_NAME"] in ['unet', 'UNET', 'UNet', 'Unet']:
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=2,
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            # channels=(32, 64, 128, 256, 512),
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            # dropout = config["DROPOUT"],
        )



    elif config["MODEL_NAME"] in ['attunet', 'ATT-UNET', 'Att-UNet', 'attUnet', 'att-unet', 'att_unet']:
        from monai.networks.nets import AttentionUnet
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            # num_res_units=2,
        )

    elif config["MODEL_NAME"] in ['D_UNet_monai', 'dunet_m', 'DUNet_monai', 'denseunet_m', 'dunet_monai']:
        from core.model.Dense_UNet_monai_base import DenseUNet
        model = DenseUNet(
            spatial_dims=2,
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            # channels=(32, 64, 128, 256, 512),
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout = config["DROPOUT"],
        )













    elif config["MODEL_NAME"] in ['dynunet', 'DynUNet', 'DynUnet']:
        from monai.networks.nets import DynUNet
        assert config["DynUnet_strides"][0] == 1, "Strides should be start with 1"
        model = DynUNet(
            spatial_dims=2,
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            kernel_size=config["DynUnet_kernel"],#[3,3,3,3,3],
            strides=config["DynUnet_strides"],#[1,2,2,2,2],
            upsample_kernel_size=config["DynUnet_upsample"],#[2,2,2,2,2],
            filters=config["DynUnet_filters"],#[64, 96, 128, 192, 256, 384, 512, 768, 1024],
            dropout=config["DROPOUT"],
            deep_supervision=True,
            deep_supr_num=len(config["DynUnet_strides"])-2,
            norm_name='INSTANCE',
            act_name='leakyrelu',
            res_block=config["DynUnet_residual"], #False
            trans_bias=False,
        )

        
    elif config["MODEL_NAME"] in ['swinUNETR', 'sunetr', 'sUNETR', 'swin']:
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=config["INPUT_SHAPE"],
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            feature_size=24,
            spatial_dims=2,
            use_checkpoint=True,
        )



    elif config["MODEL_NAME"] in ['CA_Swin', 'CA_swin', 'caswin', 'CAswin']:
        from .model.Swin_UNETR_CoorATT import CA_SwinUNETR
        model = CA_SwinUNETR(
            img_size=config["INPUT_SHAPE"],
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            feature_size=24,
            spatial_dims=2,
            use_checkpoint=True,
        )

    elif config["MODEL_NAME"] in ['CA_Swin_deep', 'CA_swin_deep', 'caswin_deep', 'CAswin_deep']:
        from .model.Swin_UNETR_CoorATT_deepsupervision import CA_SwinUNETR_deepsupervision
        model = CA_SwinUNETR_deepsupervision(
            img_size=config["INPUT_SHAPE"],
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            feature_size=24,
            spatial_dims=2,
            use_checkpoint=True,
            deep_supervision = True,
            deep_supr_num=3,
        )

    elif config["MODEL_NAME"] in ['Swin_deep', 'swin_deep', 'Swin_Deep']:
        from .model.Swin_UNETR_deepsupervision import SwinUNETR_deepsupervision
        model = SwinUNETR_deepsupervision(
            img_size=config["INPUT_SHAPE"],
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            feature_size=24,
            spatial_dims=2,
            use_checkpoint=True,
            deep_supervision = True,
            deep_supr_num=3,
        )

    elif config["MODEL_NAME"] in ['CA_Swin_deep_dense', 'CA_swin_deep_dense', 'caswin_deep_dense', 'CAswin_deep_dense']:
        from .model.Swin_UNETR_CoorATT_deepsupervision_dense import CA_SwinUNETR_deepsupervision_dense
        model = CA_SwinUNETR_deepsupervision_dense(
            img_size=config["INPUT_SHAPE"],
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            feature_size=24,
            spatial_dims=2,
            use_checkpoint=True,
            deep_supervision = True,
            deep_supr_num=3,
        )


    elif config["MODEL_NAME"] in ['CA_Swin_deep_dense_tanh', 'CA_swin_deep_dense_tanh', 'caswin_deep_dense_tanh', 'CAswin_deep_dense_tanh']:
        from .model.Swin_UNETR_CoorATT_deepsupervision_dense_tanh import CA_SwinUNETR_deepsupervision_dense
        model = CA_SwinUNETR_deepsupervision_dense(
            img_size=config["INPUT_SHAPE"],
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            feature_size=24,
            spatial_dims=2,
            use_checkpoint=True,
            deep_supervision = True,
            deep_supr_num=3,
        )









    elif config["MODEL_NAME"] in ['CADD_UNet', 'Caddunet', 'CADD_Unet', 'caddunet', 'cadd-unet']:
        from core.model.CADD_UNet import CADD_UNet
        model = CADD_UNet(
            in_channel=config["CHANNEL_IN"],
            out_channel=config["CHANNEL_OUT"],
            channel_list=[32, 64, 128, 256, 512],
            kernel_size= (3, 3),
            drop_rate = config["DROPOUT"],
        )


    elif config["MODEL_NAME"] in ['D_UNet', 'dunet', 'DUNet', 'denseunet']:
        from core.model.Dense_UNet import Dense_UNet
        model = Dense_UNet(
            in_channel=config["CHANNEL_IN"],
            out_channel=config["CHANNEL_OUT"],
            # channel_list=[32, 64, 128, 256, 512],
            channel_list=[64, 128, 256, 512, 1024],
            kernel_size= (3, 3),
            drop_rate = config["DROPOUT"],
        )


    elif config["MODEL_NAME"] in ['convnext', 'Convnext', 'ConvNext','convnext-unet', 'ConvNext-UNet']:
        from core.model.ConvNext_UNet import U_ConvNext
        model = U_ConvNext(
            img_ch=config["CHANNEL_IN"],
            output_ch=config["CHANNEL_OUT"],
            channels=24
            # channel_list=[64, 128, 256, 512, 1024],
            # kernel_size= (3, 3),
            # drop_rate = config["DROPOUT"],
        )    


    elif config["MODEL_NAME"] in ['r2att_convnext', 'R2Att_Convnext', 'R2Att_ConvNext']:
        from core.model.ConvNext_UNet import R2AttU_ConvNext
        model = R2AttU_ConvNext(
            img_ch=config["CHANNEL_IN"],
            output_ch=config["CHANNEL_OUT"],
            channels=24,
            t=2
            # channel_list=[64, 128, 256, 512, 1024],
            # kernel_size= (3, 3),
            # drop_rate = config["DROPOUT"],
        )    










        # print(model)  


    elif config["MODEL_NAME"] in ['CADD_UNet_light', 'Caddunet_light', 'cadd_unet_light', 'caddunet_light']:
        from core.model.CADD_UNet_light import CADD_UNet_light
        model = CADD_UNet_light(
            in_channel=config["CHANNEL_IN"],
            out_channel=config["CHANNEL_OUT"],
            channel_list=[32, 64, 128, 256, 512],
            kernel_size= (3, 3),
            drop_rate = config["DROPOUT"],
        )

    elif config["MODEL_NAME"] in ['DD_UNet', 'ddunet', 'DD_Unet', 'DDunet']:
        from core.model.DD_UNet import DD_UNet
        model = DD_UNet(
            in_channel=config["CHANNEL_IN"],
            out_channel=config["CHANNEL_OUT"],
            # channel_list=[32, 64, 128, 256, 512],
            channel_list=[64, 128, 256, 512, 1024],
            kernel_size= (3, 3),
            drop_rate = config["DROPOUT"],
        )
        # print(model)  # 필요없으면 지우셔도 됩니다!


    assert model is not None, 'Model Error!'    
    return model


def call_optimizer(config, model):
    if config["OPTIM_NAME"] in ['SGD', 'sgd']:
        return optim.SGD(model.parameters(), lr=config["LR_INIT"], momentum=config["MOMENTUM"])
    elif config["OPTIM_NAME"] in ['ADAM', 'adam', 'Adam']:
        return optim.Adam(model.parameters(), lr=config["LR_INIT"])
    elif config["OPTIM_NAME"] in ['ADAMW', 'adamw', 'AdamW', 'Adamw']:
        return optim.AdamW(model.parameters(), lr=config["LR_INIT"], weight_decay=config["LR_DECAY"])
    elif config["OPTIM_NAME"] in ['ADAGRAD', 'adagrad', 'AdaGrad']:
        return optim.Adagrad(model.parameters(), lr=config["LR_INIT"], lr_decay=config["LR_DECAY"])
    else:
        return None
