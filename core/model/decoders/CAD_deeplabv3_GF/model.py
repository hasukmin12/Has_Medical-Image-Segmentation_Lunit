import torch
from torch import nn
from typing import Optional

from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.encoders import get_encoder
from .decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder
from segmentation_models_pytorch.base.modules import Activation
from ..CAD_deeplabv3.model import CAD_deeplabv3Plus


class CAD_DeepLabV3Plus_GF(CAD_deeplabv3Plus):
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super(CAD_DeepLabV3Plus_GF, self).__init__(
            encoder_name=encoder_name,
            encoder_depth = encoder_depth,
            encoder_weights=encoder_weights,
            encoder_output_stride = encoder_output_stride,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            upsampling=upsampling,
            aux_params=aux_params
        )

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(
                    encoder_output_stride
                )
            )



        # Delete DeepLabV3Plus Head
        self.segmentation_head = None

        # Convert all Encoder/Decoder activations to 0  # 이거 왜 하지..?
        convert_relu_to_mish(self.encoder)
        convert_relu_to_mish(self.decoder)

        self.cellprob_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.gradflow_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=4,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )


        # self.cellprob_head = DeepSegmantationHead(
        #     in_channels=decoder_channels, out_channels=classes, kernel_size=3,
        # )
        # self.gradflow_head = DeepSegmantationHead(
        #     in_channels=decoder_channels, out_channels=4, kernel_size=3,
        # )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        gradflow_mask = self.gradflow_head(decoder_output)
        cellprob_mask = self.cellprob_head(decoder_output)

        # masks = torch.cat([gradflow_mask, cellprob_mask], dim=1)

        return cellprob_mask, gradflow_mask# masks






class DeepSegmantationHead(nn.Sequential):
    """SegmentationHead for Cell Probability & Grad Flows"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        conv2d_1 = nn.Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        bn = nn.BatchNorm2d(in_channels // 2)
        conv2d_2 = nn.Conv2d(
            in_channels // 2,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        mish = nn.Mish(inplace=True)

        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        activation = Activation(activation)
        super().__init__(conv2d_1, mish, bn, conv2d_2, upsampling, activation)



def convert_relu_to_mish(model):
    """Convert ReLU atcivation to Mish"""
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Mish(inplace=True))
        else:
            convert_relu_to_mish(child)