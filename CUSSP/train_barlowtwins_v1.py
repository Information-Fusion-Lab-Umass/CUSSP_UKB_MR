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
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as VisionF
from torchvision.datasets import CIFAR10
from torchvision import models as vision_models
from torchvision.utils import make_grid

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics import functional as tmf
from sklearn import metrics as sk_metrics
from sklearn.model_selection import train_test_split


from BarlowTwinsMR.BarlowTwins.models import BarlowTwins
from BarlowTwinsMR.BarlowTwins.models import BarlowTwins_v1
from BarlowTwinsMR.BarlowTwins.models import ProjectionHead
from BarlowTwinsMR.BarlowTwins.models import OnlineFineTuner

from BarlowTwinsMR.BarlowTwins.dataset import LAX_4Ch_patch_Dataset
from BarlowTwinsMR.BarlowTwins.dataset import lax_4ch_normalization
from BarlowTwinsMR.BarlowTwins.dataset import get_mean_and_std

from BarlowTwinsMR.BarlowTwins.transforms import ToTensorTransform
from BarlowTwinsMR.BarlowTwins.transforms import BarlowTwinsTransform

from BarlowTwinsMR.BarlowTwins.loss import BarlowTwinsLoss
from BarlowTwinsMR.BarlowTwins.encoders import get_encoder

def freeze_model(model):
    print("freezing model layers")
    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.encoder.layer4[1].parameters():
        param.requires_grad = True

    return model

def evaluate_model(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = model.encoder.to(device)
    online_finetuner = model.online_finetuner.to(device)

    out_df = pd.DataFrame()
    for batch in dataloader:
        img, label = batch
        prob = online_finetuner(encoder(img.to(device)).detach())
        preds = F.softmax(prob, dim=1).argmax(dim=1).cpu().numpy().astype(int)
        for _label, _pred in zip(label.cpu().numpy().astype(int), preds):
            out_df = out_df.append({"label": _label, "pred": _pred}, ignore_index=True)

    return out_df

def get_datasets(args, figsize=(12,8)):
    """
    Prepare train_csv, val_csv
    """

    #label_csv = pd.read_csv(f"{args.data_dir}/labels.csv")
    label_csv = pd.read_csv(args.label_csv)
    
    train_ID, val_ID, train_label, val_label = train_test_split(
                        label_csv.PID, label_csv.LABEL, test_size=0.2, random_state=1234
                        )
    
    train_csv = pd.DataFrame({"PID":train_ID, "LABEL": train_label}).reset_index(drop=True)
    val_csv = pd.DataFrame({"PID":val_ID, "LABEL": val_label}).reset_index(drop=True)

    """
    Pack data into dataloaders
    """
    patch_mean, patch_std = get_mean_and_std(label_csv.PID, args.patch_type, args.data_dir)
    
    
    lax_4ch_normalize = lax_4ch_normalization(patch_mean, patch_std)
    train_transform = BarlowTwinsTransform(
        train=True, input_height=32, 
        perspective=True, gaussian_blur=False, 
        normalize=lax_4ch_normalize
    )
    
    val_transform = BarlowTwinsTransform(
        train=False, input_height=32, 
        disable_augmentation=True,
        normalize=lax_4ch_normalize
    )
    
    train_dataset = LAX_4Ch_patch_Dataset(train_csv, f"{args.data_dir}/{args.patch_type}",
                                          transform=train_transform)
    val_dataset = LAX_4Ch_patch_Dataset(val_csv, f"{args.data_dir}/{args.patch_type}",
                                        transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    
    print(f"datapoint shape: {train_dataset[0][0][0].shape}")


    if args.verbose:
        for batch in val_loader:
            (img1, img2, _), label = batch
            break

        print(img1.shape)
        img_grid1 = make_grid(img1[:,0:1], normalize=True)
        img_grid2 = make_grid(img2[:,0:1], normalize=True)


        def show(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
            for i, img in enumerate(imgs):
                img = img.detach()
                img = VisionF.to_pil_image(img)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        show(img_grid1)
        show(img_grid2)

    return train_loader, val_loader


def train(args):
    print(f"Cuda is available: {torch.cuda.is_available()}")
    train_loader, val_loader = get_datasets(args)

    """
    Make the model
    """
    encoder_kwargs = dict(pretrained=args.pretrained)
    encoder = get_encoder(args.encoder, **encoder_kwargs)

    encoder_output_dim = encoder(torch.rand(1,50,32,32)).shape[1]
    #print(f"Encoder output dimension: {encoder_output_dim}") 
    print(f"BarlowTwins dimensions : 50x32x32 -> {encoder_output_dim} -> {args.z_dim}")

    online_finetuner = OnlineFineTuner(encoder_output_dim=encoder_output_dim, 
                                       dropout=args.dropout, 
                                       num_classes=args.num_classes, 
                                       classifier=args.classifier,
                                       learning_rate=args.cls_learning_rate,
                                       weight_decay=args.cls_weight_decay)
    checkpoint_callback = ModelCheckpoint(dirpath=args.out_dir,
                                          every_n_epochs=min(args.every_n_epochs, args.max_epochs),
                                          save_top_k=-1, save_last=True)

    model_kwargs = dict(
        encoder=encoder,
        encoder_output_dim=encoder_output_dim,
        num_training_samples=len(train_loader.dataset),
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        online_finetuner=online_finetuner.classifier
        )

    model_kwargs['online_finetuner'] = [512, 32, 2]
    model = BarlowTwins_v1(**model_kwargs)
    model = model.load_from_checkpoint(checkpoint_path=args.encoder_path, strict=False, **model_kwargs)
    model.reset_online_finetuner(online_finetuner.classifier)
    model_kwargs['online_finetuner'] = online_finetuner.classifier


    if args.online_classifier:
        callbacks=[online_finetuner, checkpoint_callback]
    else:
        callbacks=[checkpoint_callback]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        gpus=torch.cuda.device_count(),
        precision=32,
        callbacks=callbacks,
        log_every_n_steps = 4,
        default_root_dir=args.out_dir,
    )

    if args.freeze_model:
        model = freeze_model(model)

    # train the model
    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    print(">>>>>> Best model saved at ", best_model_path)
    model = model.load_from_checkpoint(checkpoint_path=best_model_path, **model_kwargs)
    
    """
    Evaluate the model on the train/val dataset
    """
    if args.online_classifier:
        downstream_train_dataset = LAX_4Ch_patch_Dataset(train_loader.dataset.img_labels, 
                                                         f"{args.data_dir}/{args.patch_type}",
                                                         transform=val_loader.dataset.transform.finetune_transform)
        downstream_train_loader = DataLoader(downstream_train_dataset, batch_size=4, shuffle=False)
        train_df = evaluate_model(model, downstream_train_loader)

        downstream_val_dataset = LAX_4Ch_patch_Dataset(val_loader.dataset.img_labels, 
                                                       f"{args.data_dir}/{args.patch_type}",
                                                       transform=val_loader.dataset.transform.finetune_transform)
        downstream_val_loader = DataLoader(downstream_val_dataset, batch_size=4, shuffle=False)
        val_df = evaluate_model(model, downstream_val_loader)
        
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)

        train_df.to_csv(f"{args.out_dir}/train_preds.csv", index=False)
        val_df.to_csv(f"{args.out_dir}/val_preds.csv", index=False)



def parse_args(default=False):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--batch_size", default=32, type=int, help="Batch size used to train the model.")
    argparser.add_argument("--num_workers", default=4, type=int, help="Number of workers for dataloaders.")
    argparser.add_argument("--label_csv", default="/home/kexiao/Data/mr_oc/mr/patches/labels.csv", type=str, help="Path to the labels csv file")
    argparser.add_argument("--data_dir", default="/home/kexiao/Data/mr_oc/mr/patches/npy64/anchor", type=str, help="Path to the data folder")
    argparser.add_argument("--out_dir", default="outputs", type=str, help="The folder to save output to.")
    argparser.add_argument("--encoder_path", default="lightning_logs/model/last.ckpt", type=str, help="The folder to load checkpointed model from.")
    argparser.add_argument("--encoder", default="resnet18", type=str, help="Encoder used from torchvision library.")
    argparser.add_argument("--patch_type", default="la", type=str, help="The patch type: la/pa/og")
    argparser.add_argument("--warmup_epochs", default=100, type=int, help="The number of epochs to use for warmup")
    argparser.add_argument("--max_epochs", default=200, type=int, help="The max number of epochs to train.")
    argparser.add_argument("--every_n_epochs", default=20, type=int, help="The interval to checkpoint the model with.")
    argparser.add_argument("--z_dim", default=2048, type=int, help="The z_dim used by the projection head.")
    argparser.add_argument("--dropout", default=0.5, type=float, help="The dropout probability for classifier.")
    argparser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate of BarlowTwins.")
    argparser.add_argument("--weight_decay", default=0.0, type=float, help="The weight decay of BarlowTwins.")
    argparser.add_argument("--cls_learning_rate", default=1e-4, type=float, help="The learning rate of the classifier.")
    argparser.add_argument("--cls_weight_decay", default=1e-3, type=float, help="The weight decay of classifier.")
    argparser.add_argument("--num_classes", default=2, type=int, help="The number of classes in the classification task.")
    argparser.add_argument("--verbose", action="store_true", help="Whether to print out debugging messages.")
    argparser.add_argument("--pretrained", action="store_true", help="Whether to load pre-trained weights from torchvision models into the encoder.")
    argparser.add_argument('--classifier', type=int, default=[], nargs='+', help='Number of nodes in the layers of the linear classifier for online tuning.')
    argparser.add_argument("--online_classifier", action="store_true", help="Whether to train the classifier online with training of BarlowTwins.")
    argparser.add_argument("--freeze_model", action="store_true", help="Whether to freeze the parameters in the encoder except for the last layer.")

    if default:
        args = argparser.parse_args("")
    else:
        args = argparser.parse_args()

    return args

def main():
    args = parse_args()
    train(args)



if __name__ == '__main__':
    main()
