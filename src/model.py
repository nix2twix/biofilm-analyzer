# src/model.py
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

def build_model(encoderName = 'resnet34', encoderWeights = 'imagenet', activation = None):
    model = smp.UnetPlusPlus(
        encoder_name=encoderName,
        encoder_weights=encoderWeights,
        in_channels=1,  # grayscale
        classes=2,
        activation=activation
    )
    return model

