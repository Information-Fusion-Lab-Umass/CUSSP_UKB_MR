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
## Dataset
#####################################################


def label_edit(label):
    if label > 0:
        return 1
    else:
        return 0
    

def chunk_sequence(sequence, es_frame, frame_window):
    if isinstance(sequence, list) or isinstance(sequence, tuple):
        return [chunk_sequence(sub_sequence, es_frame, frame_window) for sub_sequence in sequence]
    else:
        return sequence[es_frame - frame_window : es_frame]

class LAX_4Ch_patch_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=label_edit,
                 data_dtype=np.float32, es_frame_file=None, frame_window=50):
        if isinstance(annotations_file, str):
            self.img_labels = pd.read_csv(annotations_file)
        else:
            self.img_labels = annotations_file
            
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_dtype = data_dtype

        self.frame_window = frame_window
        if es_frame_file is not None:
            es_frame_df = pd.read_csv(es_frame_file)
            self.img_labels = self.img_labels.merge(es_frame_df, on='PID')
            self.chunk_frames = True
        elif frame_window < 50:
            self.img_labels['ES_frame'] = [frame_window] * len(self.img_labels)
            self.chunk_frames = True
        else:
            self.chunk_frames = False

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        data_path = os.path.join(self.img_dir, f"{self.img_labels.PID[idx]}.npy")
        sequence = np.load(data_path).astype(self.data_dtype)
        #sequence = sequence / sequence.max() * 255.0

        
        label = self.img_labels.LABEL[idx]
        if self.transform:
            sequence = self.transform(sequence)
        if self.target_transform:
            label = self.target_transform(label)
            
        if self.chunk_frames:
            es_frame = self.img_labels.ES_frame[idx]
            sequence = chunk_sequence(sequence, es_frame, self.frame_window)

        return sequence, label


def get_pair_index(total_n_points, idx):
    pair_idx = np.random.randint(total_n_points)
    while idx == pair_idx:
        pair_idx = np.random.randint(total_n_points)

    return pair_idx


class LAX_4Ch_Siamese_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, 
                 transform=None, target_transform=label_edit,
                 data_dtype=np.float32, es_frame_file=None, frame_window=50):

        if isinstance(annotations_file, str):
            self.img_labels = pd.read_csv(annotations_file)
        else:
            self.img_labels = annotations_file

        self.negative_df = self.img_labels.query("LABEL==0").reset_index(drop=True)
            
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_dtype = data_dtype

        self.frame_window = frame_window
        if es_frame_file is not None:
            es_frame_df = pd.read_csv(es_frame_file)
            self.img_labels = self.img_labels.merge(es_frame_df, on='PID')
            self.chunk_frames = True
        elif frame_window < 50:
            self.img_labels['ES_frame'] = [frame_window] * len(self.img_labels)
            self.chunk_frames = True
        else:
            self.chunk_frames = False

    def __len__(self):
        return len(self.negative_df)

    def __get_single_item__(self, idx, negative=False):
        if negative:
            data_path = os.path.join(self.img_dir, f"{self.negative_df.PID[idx]}.npy")
            sequence = np.load(data_path).astype(self.data_dtype)
            label = self.negative_df.LABEL[idx]
        else:
            data_path = os.path.join(self.img_dir, f"{self.img_labels.PID[idx]}.npy")
            sequence = np.load(data_path).astype(self.data_dtype)
            label = self.img_labels.LABEL[idx]


        if self.transform:
            sequence = self.transform(sequence)
        if self.target_transform:
            label = self.target_transform(label)

        if self.chunk_frames:
            es_frame = self.img_labels.ES_frame[idx]
            sequence = chunk_sequence(sequence, es_frame, self.frame_window)

        return sequence, label

    def __getitem__(self, idx):
        mapped_idx = self.img_labels.query(f"PID=={self.negative_df.PID[idx]}").index[0]
        idx_pair = get_pair_index(len(self.img_labels), mapped_idx)

        x1, y1 = self.__get_single_item__(idx, negative=True)
        x2, y2 = self.__get_single_item__(idx_pair, negative=False)

        return (x1,y1), (x2,y2)




def load_lax_4ch_patch(pid, data_dir):
    data = np.load(f"{data_dir}/{pid}.npy")
    return data


def get_mean_and_std(pids, patch_type, data_dir, num_frames=50):
    n_datapoints = len(pids)
    running_mean = np.zeros(num_frames).astype(float)
    for _pid in pids:
        data = load_lax_4ch_patch(_pid, f"{data_dir}/{patch_type}")
        running_mean += data.mean(axis=(1,2)) / n_datapoints
    
    running_var = np.zeros(num_frames).astype(float)
    for _pid in pids:
        data = load_lax_4ch_patch(_pid, f"{data_dir}/{patch_type}")
        running_var += np.mean((data - running_mean[:,None,None]) ** 2, axis=(1,2)) / n_datapoints
    running_std = np.sqrt(running_var)
    
    return running_mean, running_std


def lax_4ch_normalization(data_mean, data_std):
    normalize = transforms.Normalize(
        mean=data_mean, std=data_std
    )
    return normalize





