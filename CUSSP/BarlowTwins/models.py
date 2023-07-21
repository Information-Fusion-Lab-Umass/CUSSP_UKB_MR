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

from .loss import BarlowTwinsLoss
from .loss import ContrastiveLoss

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

def create_linear_classifier(layers, dropout=0.4):
    cls = []
    for i in range(len(layers)-2):
        cls.append(nn.Linear(layers[i], layers[i+1]))
        cls.append(nn.ReLU())
        cls.append(nn.Dropout(p=dropout))

    cls.append(nn.Linear(layers[-2], layers[-1]))

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
        weight_decay=0.0,
        dropout=0.5,
        warmup_epochs=100,
        max_epochs=200,
        online_finetuner=[],
        **kwargs
    ):
        super().__init__()

        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=encoder_output_dim, hidden_dim=z_dim, output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.dropout = dropout

        self.train_iters_per_epoch = num_training_samples // batch_size
        self.classifier_layers = online_finetuner
        self.online_finetuner = create_linear_classifier(self.classifier_layers, self.dropout)

    def reset_online_finetuner(self, classifier):
        self.online_finetuner = create_linear_classifier(classifier, self.dropout)
        self.classifier = create_linear_classifier(classifier, self.dropout)


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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

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

class BarlowTwins_v1(BarlowTwins):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.classifier = create_linear_classifier(self.classifier_layers, self.dropout)
        class_weight = torch.tensor([0.15, 0.85]).to(self.device)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=class_weight)
    
    
    def shared_step(self, batch):
        (x1, x2, x3), y = batch
        
        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))
        cc_loss = self.loss_fn(z1, z2)
        
        preds = self.classifier(self.encoder(x3))
        cls_loss = self.ce_loss(preds, y)
        
        loss = cc_loss * .1 + cls_loss * .9
        
        return loss
        
        
class BarlowTwins_Siamese(BarlowTwins):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.classifier = create_linear_classifier(self.classifier_layers, self.dropout)
        class_weight = torch.tensor([0.15, 0.85]).to(self.device)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=class_weight)

        self.contrastive_loss = ContrastiveLoss(margin=1.0)
        self.cc_w, self.cls_w, self.ct_w = kwargs.get("loss_weights")
        # cross correlation, classification, contrastive
    
    def shared_step(self, batch):
        (x1, y1), (x2, y2) = batch
        
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)

        z1 = self.projection_head(emb1)
        z2 = self.projection_head(emb2)

        cc_loss = self.loss_fn(z1, z2)
        
        preds = self.classifier(self.encoder(x2))
        cls_loss = self.ce_loss(preds, y2)


        y = torch.abs(y1 - y2)
        ct_loss = self.contrastive_loss(emb1, emb2, y)

        #loss = cc_loss * .1 + cls_loss * .5 + .4 * ct_loss
        #loss = ct_loss
        loss = self.cc_w * cc_loss + self.cls_w * cls_loss + self.ct_w * ct_loss
        
        return loss


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=2048):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
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
        dropout: float,
        num_classes: int,
        classifier: list,
        weight_decay: float,
        learning_rate: float,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.classifier = [encoder_output_dim] + classifier + [num_classes]
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.train_state = {"labels":[], "preds":[], "probs":[]}
        self.validation_state = {"labels":[], "preds":[], "probs":[]}
        self.best_f1 = 0.0

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        # add linear_eval layer and optimizer
        #pl_module.online_finetuner = nn.Linear(self.encoder_output_dim, self.num_classes).to(pl_module.device)
        pl_module.online_finetuner = create_linear_classifier(self.classifier, self.dropout).to(pl_module.device)

        self.optimizer = torch.optim.Adam(pl_module.online_finetuner.parameters(),
                                          lr=self.learning_rate, weight_decay=self.weight_decay)

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

        class_weight = torch.tensor([0.15, 0.85]).to(pl_module.device)
        loss = F.cross_entropy(preds, y, weight=class_weight)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        acc = tmf.accuracy(F.softmax(preds, dim=1), y)
        
        y_true = y.cpu().numpy()
        y_prob = F.softmax(preds, dim=1).detach().cpu().numpy()
        y_pred = y_prob.argmax(axis=1)

        prec, rec, f1, _ = sk_metrics.precision_recall_fscore_support(y_true, y_pred, 
                                                                     average='binary',
                                                                     zero_division=0)

        pl_module.log("online_train_f1", f1, on_step=True, on_epoch=False)
        pl_module.log("online_train_precision", prec, on_step=True, on_epoch=False)
        pl_module.log("online_train_recall", rec, on_step=True, on_epoch=False)
        
        pl_module.log("online_train_acc", acc, on_step=True, on_epoch=False)
        pl_module.log("online_train_CEloss", loss, on_step=True, on_epoch=False)

        self.train_state["labels"].append(y_true)
        self.train_state["preds"].append(y_pred)
        self.train_state["probs"].append(y_prob)

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

        class_weight = torch.tensor([0.15, 0.85]).to(pl_module.device)
        loss = F.cross_entropy(preds, y, weight=class_weight)

        acc = tmf.accuracy(F.softmax(preds, dim=1), y)
        
        y_true = y.cpu().numpy()
        y_prob = F.softmax(preds, dim=1).detach().cpu().numpy()
        y_pred = y_prob.argmax(axis=1)

        prec, rec, f1, _ = sk_metrics.precision_recall_fscore_support(y_true, y_pred, 
                                                                     average='binary', 
                                                                     zero_division=0)
        
        pl_module.log("online_val_f1", f1, on_step=False, on_epoch=True)
        pl_module.log("online_val_precision", prec, on_step=False, on_epoch=True)
        pl_module.log("online_val_recall", rec, on_step=False, on_epoch=True)
        
        pl_module.log("online_val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_CEloss", loss, on_step=False, on_epoch=True, sync_dist=True)

        self.validation_state["labels"].append(y_true)
        self.validation_state["preds"].append(y_pred)
        self.validation_state["probs"].append(y_prob)


    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
        ):
        y_true = np.concatenate(self.train_state["labels"])
        y_pred = np.concatenate(self.train_state["preds"])
        y_prob = np.concatenate(self.train_state["probs"])

        acc = sk_metrics.accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = sk_metrics.precision_recall_fscore_support(y_true, y_pred, 
                                                                     average='binary', 
                                                                     zero_division=0)
        auc = sk_metrics.roc_auc_score(y_true, y_prob[:,1])
        ap = sk_metrics.average_precision_score(y_true, y_prob[:,1])

        pl_module.log("train_acc", acc, on_step=False, on_epoch=True)
        pl_module.log("train_f1", f1, on_step=False, on_epoch=True)
        pl_module.log("train_prec", prec, on_step=False, on_epoch=True)
        pl_module.log("train_rec", rec, on_step=False, on_epoch=True)
        pl_module.log("train_auc", auc, on_step=False, on_epoch=True)
        pl_module.log("train_ap", ap, on_step=False, on_epoch=True)

        self.train_df = pd.DataFrame({"label": list(y_true), 
                                      "pred": list(y_pred),
                                      "prob": list(y_prob[:,1])})
        self.train_state["labels"] = []
        self.train_state["preds"] = []
        self.train_state["probs"] = []


    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
        ):
        y_true = np.concatenate(self.validation_state["labels"])
        y_pred = np.concatenate(self.validation_state["preds"])
        y_prob = np.concatenate(self.validation_state["probs"])

        acc = sk_metrics.accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = sk_metrics.precision_recall_fscore_support(y_true, y_pred, 
                                                                     average='binary', 
                                                                     zero_division=0)
        auc = sk_metrics.roc_auc_score(y_true, y_prob[:,1])
        ap = sk_metrics.average_precision_score(y_true, y_prob[:,1])

        pl_module.log("validation_acc", acc, on_step=False, on_epoch=True)
        pl_module.log("validation_f1", f1, on_step=False, on_epoch=True)
        pl_module.log("validation_prec", prec, on_step=False, on_epoch=True)
        pl_module.log("validation_rec", rec, on_step=False, on_epoch=True)
        pl_module.log("validation_auc", auc, on_step=False, on_epoch=True)
        pl_module.log("validation_ap", ap, on_step=False, on_epoch=True)

        self.val_df = pd.DataFrame({"label": list(y_true), 
                                    "pred": list(y_pred),
                                    "prob": list(y_prob[:,1])})
        self.validation_state["labels"] = []
        self.validation_state["preds"] = []
        self.validation_state["probs"] = []

        if f1 > self.best_f1:
            if not os.path.isdir(trainer.default_root_dir):
                os.makedirs(trainer.default_root_dir)
            if hasattr(self, 'train_df'):
                self.train_df.to_csv(f"{trainer.default_root_dir}/best_train_preds.csv", index=False)
            self.val_df.to_csv(f"{trainer.default_root_dir}/best_val_preds.csv", index=False)
            self.best_f1 = f1

class OnlineFineTuner_Siamese(OnlineFineTuner):
    def extract_online_finetuning_view(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #(_, _, finetune_view), y = batch
        (x1, y1), (x2, y2) = batch

        finetune_view = x2.to(device)
        y = y2.to(device)

        return finetune_view, y
