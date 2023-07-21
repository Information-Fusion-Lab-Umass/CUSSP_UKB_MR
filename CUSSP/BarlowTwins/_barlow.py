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
## Model
#####################################################

def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0

def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)

def create_linear_classifier(layers):
    cls = []
    for i in range(len(layers)-1):
        cls.append(nn.Linear(layers[i], layers[i+1]))

    return nn.Sequential(*cls)

class BarlowTwins(LightningModule):
    def __init__(
        self,
        encoder,
        encoder_output_dim,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=128,
        learning_rate=1e-4,
        warmup_epochs=100,
        max_epochs=200,
        online_finetuner=[],
    ):
        super().__init__()

        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=encoder_output_dim, hidden_dim=encoder_output_dim, output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.train_iters_per_epoch = num_training_samples // batch_size
        self.online_finetuner = create_linear_classifier(online_finetuner)

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        (x1, x2, _), _ = batch

        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))

        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=2048):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)


class OnlineFineTuner(Callback):
    def __init__(
        self,
        encoder_output_dim: int,
        num_classes: int,
        classifier: list,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        self.classifier = [encoder_output_dim] + classifier + [num_classes]

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        # add linear_eval layer and optimizer
        #pl_module.online_finetuner = nn.Linear(self.encoder_output_dim, self.num_classes).to(pl_module.device)
        pl_module.online_finetuner = create_linear_classifier(self.classifier).to(pl_module.device)

        self.optimizer = torch.optim.Adam(pl_module.online_finetuner.parameters(), lr=1e-4)

    def extract_online_finetuning_view(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (_, _, finetune_view), y = batch
        finetune_view = finetune_view.to(device)
        y = y.to(device)

        return finetune_view, y

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        acc = tmf.accuracy(F.softmax(preds, dim=1), y)
        
        y_pred = F.softmax(preds, dim=1).argmax(dim=1).cpu().numpy()
        f1 = sk_metrics.f1_score(y.cpu().numpy(), y_pred, zero_division=0)
        prec = sk_metrics.precision_score(y.cpu().numpy(), y_pred, zero_division=0)
        rec = sk_metrics.recall_score(y.cpu().numpy(), y_pred, zero_division=0)
        pl_module.log("online_train_f1", f1, on_step=True, on_epoch=False)
        pl_module.log("online_train_precision", prec, on_step=True, on_epoch=False)
        pl_module.log("online_train_recall", rec, on_step=True, on_epoch=False)
        
        pl_module.log("online_train_acc", acc, on_step=True, on_epoch=False)
        pl_module.log("online_train_loss", loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        acc = tmf.accuracy(F.softmax(preds, dim=1), y)
        
        y_pred = F.softmax(preds, dim=1).argmax(dim=1).cpu().numpy()
        f1 = sk_metrics.f1_score(y.cpu().numpy(), y_pred, zero_division=0)
        prec = sk_metrics.precision_score(y.cpu().numpy(), y_pred, zero_division=0)
        rec = sk_metrics.recall_score(y.cpu().numpy(), y_pred, zero_division=0)
        
        pl_module.log("online_val_f1", f1, on_step=False, on_epoch=True)
        pl_module.log("online_val_precision", prec, on_step=False, on_epoch=True)
        pl_module.log("online_val_recall", rec, on_step=False, on_epoch=True)
        
        pl_module.log("online_val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_f1", f1, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)



#####################################################
## Loss
#####################################################

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=512):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag

#####################################################
## Transform
#####################################################
#class BarlowTwinsTransform:
#    def __init__(self, train=True, input_height=224, gaussian_blur=True, normalize=None):
#
#        self.input_height = input_height
#        self.gaussian_blur = gaussian_blur
#        self.normalize = normalize
#        self.train = train
#
#        #color_jitter = transforms.ColorJitter(
#        #    0.8 * self.jitter_strength,
#        #    0.8 * self.jitter_strength,
#        #    0.8 * self.jitter_strength,
#        #    0.2 * self.jitter_strength,
#        #)
#
#        #color_transform = [transforms.RandomApply([color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2)]
#        random_transform = [transforms.RandomPerspective(distortion_scale=0.2, p=0.4)]
#        
#        random_transform.append(transforms.RandomApply([
#            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.8, 0.95))],
#            p=0.4
#        ))
#
#        if self.gaussian_blur:
#            kernel_size = int(0.1 * self.input_height)
#            if kernel_size % 2 == 0:
#                kernel_size += 1
#
#            random_transform.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))
#
#        self.random_transform = transforms.Compose(random_transform)
#
#        if normalize is None:
#            self.final_transform = ToTensorTransform()
#        else:
#            self.final_transform = transforms.Compose([ToTensorTransform(), normalize])
#
#        self.transform = transforms.Compose(
#            [
#                ToTensorTransform(),
#                transforms.RandomResizedCrop(self.input_height),
#                transforms.RandomHorizontalFlip(p=0.3),
#                transforms.RandomVerticalFlip(p=0.3),
#                self.random_transform,
#                self.final_transform,
#            ]
#        )
#
#        self.finetune_transform = None
#        if self.train:
#            self.finetune_transform = transforms.Compose(
#                [
#                    ToTensorTransform(),
#                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
#                    #transforms.RandomHorizontalFlip(),
#                    #transforms.RandomVerticalFlip(),
#                ]
#            )
#        else:
#            self.finetune_transform = ToTensorTransform()
#    
#    def __call__(self, sample):
#        x1 = self.transform(sample)
#        x2 = self.transform(sample)
#        x3 = self.finetune_transform(sample)
#        
#        return x1, x2, x3
#
#class ToTensorTransform:
#    def __call__(self, sample):
#        if not isinstance(sample, torch.Tensor):
#            return torch.as_tensor(sample)
#        else:
#            return sample



#####################################################
## Dataset
#####################################################


def label_edit(label):
    if label > 0:
        return 1
    else:
        return 0
    
class LAX_4Ch_patch_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=label_edit,
                 data_dtype=np.float32):
        if isinstance(annotations_file, str):
            self.img_labels = pd.read_csv(annotations_file)
        else:
            self.img_labels = annotations_file
            
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_dtype = data_dtype

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
        return sequence, label


def load_lax_4ch_patch(pid, data_dir):
    data = np.load(f"{data_dir}/{pid}.npy")
    return data


def get_mean_and_std(pids, patch_type, data_dir):
    n_datapoints = len(pids)
    running_mean = np.zeros(50).astype(float)
    for _pid in pids:
        data = load_lax_4ch_patch(_pid, f"{data_dir}/{patch_type}")
        running_mean += data.mean(axis=(1,2)) / n_datapoints
    
    running_var = np.zeros(50).astype(float)
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





