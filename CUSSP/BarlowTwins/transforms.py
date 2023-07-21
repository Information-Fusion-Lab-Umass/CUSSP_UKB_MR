from functools import partial
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
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
from torchvision.models import resnet
from torchvision.utils import make_grid

import os
import pandas as pd
from torch.utils.data import Dataset


#####################################################
## Transform
#####################################################
class BarlowTwinsTransform_OLD:
    def __init__(self, train=True, input_height=224, gaussian_blur=True, normalize=None):

        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize
        self.train = train

        random_transform = [transforms.RandomPerspective(distortion_scale=0.2, p=0.4)]
        random_transform.append(random_apply(
                                    transforms.RandomAffine(degrees=5, 
                                                            translate=(0.05, 0.05), 
                                                            scale=(0.8, 0.95)),
                                    p=0.4
                                )
        )

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            random_transform.append(random_apply(
                                        transforms.GaussianBlur(kernel_size=kernel_size), 
                                        p=0.5
                                    )
            )

        self.random_transform = transforms.Compose(random_transform)

        if normalize is None:
            self.final_transform = ToTensorTransform()
        else:
            self.final_transform = transforms.Compose([ToTensorTransform(), normalize])

        self.transform = transforms.Compose(
            [
                ToTensorTransform(),
                transforms.RandomResizedCrop(self.input_height),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                self.random_transform,
                self.final_transform,
            ]
        )

        self.finetune_transform = None
        if self.train:
            self.finetune_transform = transforms.Compose(
                [
                    ToTensorTransform(),
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomVerticalFlip(),
                ]
            )
        else:
            self.finetune_transform = ToTensorTransform()
    
    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        x3 = self.finetune_transform(sample)
        
        return x1, x2, x3


def random_apply(transform, p=0.5):
    return transforms.RandomApply([transform], p=p)

class BarlowTwinsTransform:
    def __init__(self, train=True, input_height=224, perspective=True, gaussian_blur=True, 
                 normalize=None, disable_augmentation=False):

        self.train = train
        self.input_height = input_height
        self.perspective = perspective
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.transforms =   [ToTensorTransform()]

        if disable_augmentation:
            self.transforms.append(transforms.CenterCrop(size=(32,32)))
        else:
            if perspective:
                self.transforms.append(transforms.RandomPerspective(distortion_scale=0.1, p=0.4))


            self.transforms += [random_apply(
                                    transforms.RandomAffine(degrees=(-5, 5), 
                                                            translate=(0.1, 0.1),
                                                            scale=(0.9, 1.1)),
                                    p=0.5
                            ),
                            transforms.CenterCrop(size=(37,37)),
                            transforms.RandomCrop(size=(32,32)),
                            ]

            if self.gaussian_blur:
                kernel_size = int(0.1 * self.input_height)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                self.transforms.append(random_apply(
                                            transforms.GaussianBlur(kernel_size=kernel_size), 
                                            p=0.5
                                  )
                )


        if self.train:
            self.finetune_transform = [
                    ToTensorTransform(),
                    random_apply(
                            transforms.RandomAffine(degrees=(-3, 3), 
                                                    translate=(0.06, 0.06),
                                                    scale=(0.95, 1.05)),
                            p=0.5
                        ),
                    transforms.CenterCrop(size=(34,34)),
                    transforms.RandomCrop(size=(32,32)),
                ]
        else:
            self.finetune_transform = [
                    ToTensorTransform(),
                    transforms.CenterCrop(size=(32,32)),
                ]

        if self.normalize:
            self.transforms.append(normalize)
            self.finetune_transform.append(normalize)

        self.transform = transforms.Compose(self.transforms)
        self.finetune_transform = transforms.Compose(self.finetune_transform)
    
    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        x3 = self.finetune_transform(sample)
        
        return x1, x2, x3

class SiameseTransform(BarlowTwinsTransform):
    def __call__(self, sample):
        x = self.transform(sample)
        
        return x

class ToTensorTransform:
    def __call__(self, sample):
        if not isinstance(sample, torch.Tensor):
            return torch.as_tensor(sample)
        else:
            return sample



