from functools import partial
from typing import Sequence, Tuple, Union

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as VisionF
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.metrics.functional import accuracy
from torchmetrics import functional as tmf
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import models as vision_models
from torchvision.utils import make_grid


def get_encoder(encoder_name, num_frames=50, **kwargs):
    """
    Make the model
    """
    encoder = getattr(vision_models, encoder_name)(**kwargs)


    if 'resnet' in encoder_name:
        # for CIFAR10, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool
        encoder.conv1 = nn.Conv2d(num_frames, 64, kernel_size=3, stride=1, padding=1, bias=False)
        encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        
        # replace classification fc layer of Resnet to obtain representations from the backbone
        encoder.fc = nn.Identity()
    elif 'densenet' in encoder_name:
        encoder.features.conv0 = nn.Conv2d(num_frames, 64, kernel_size=3, stride=1, padding=1, bias=False)
        encoder.features.pool0 = nn.MaxPool2d(kernel_size=1, stride=1)
        encoder.classifier = nn.Identity()
    elif 'alexnet' in encoder_name or 'vgg' in encoder_name:
        encoder.features[0] = nn.Conv2d(num_frames, 64, kernel_size=3, stride=1, padding=1, bias=False)
        encoder.features[2] = nn.MaxPool2d(kernel_size=1, stride=1)
        encoder.classifier = nn.Identity()
    else:
        raise NotImplementedError

    return encoder

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-M', '--model', type=str, help='name of the model to test output size with.')
    args = argparser.parse_args()

    x = torch.rand(1,50,32,32)

    m = get_encoder(args.model)
    encoder_out_dim = m(x).shape[1:]

    print(f"encoder output size: {encoder_out_dim}")
