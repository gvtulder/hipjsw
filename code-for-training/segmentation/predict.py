"""
Script to predict segmentations given a trained model.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

print("Waking up.")

import argparse
import numpy as np
import h5py
import hdf5plugin
import json
import os
import tqdm
import types

import torch
import torch.nn
import lightning.pytorch as pl
import monai

from dataset import datasets
from model import models

# import patch_fsspec

# on the cluster: limit the number of CPU threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# load experiment settings
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', metavar='CHECKPOINT', required=True,
                    help='checkpoint')
parser.add_argument('--model-args', metavar='JSON', required=True,
                    help='argument file for the model')

parser.add_argument('--data', metavar='HDF5', required=True, nargs='+',
                    help='dataset file (HDF5)')
parser.add_argument('--data-split', metavar='CSV', required=True, nargs='+',
                    help='list of subject IDs with subsets')
parser.add_argument('--allow-missing-samples', action='store_true',
                    help='do not raise errors on missing samples')
parser.add_argument('--predict-subset', metavar='SUBSET', nargs='+', default=['test'],
                    help='subsets to predict')
parser.add_argument('--dataset', metavar='CLASS',
                    choices=list(datasets),
                    help='dataset for task')

parser.add_argument('--output', metavar='HDF5',
                    help='output file to save segmentations')

parser.add_argument('--device', metavar='DEVICE', default='cpu',
                    help='run on CPU or GPU?')

args = parser.parse_args()
vargs = vars(args)


# load configuration
with open(args.model_args, 'r') as f:
    model_args = json.load(f)
model_args = types.SimpleNamespace(**model_args)


# set numpy, torch, python.random seeds
pl.seed_everything(model_args.seed, workers=True)


# model
print('Preparing model')
model_params = {}
model = models[model_args.model](lr=model_args.lr, num_classes=model_args.num_classes, **model_params)

# load checkpoint
print('Load checkpoint')
checkpoint = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
if args.device:
    model = model.to(args.device)

if args.output is not None:
    with h5py.File(args.output, 'w') as h5_out:
        pass


# load the train/validation split
subject_ids = {}
for filename in args.data_split:
    with open(filename, 'r') as f:
        for line in f:
            subject, subset_key = line.strip().split(',')
            if subset_key in args.predict_subset:
                if subset_key not in subject_ids:
                    subject_ids[subset_key] = []
                subject_ids[subset_key].append(subject)

print('Processing datasets')
results = []
for subset_key in subject_ids:
    ds = datasets[args.dataset or model_args.dataset](
        filenames=args.data,
        subjects=subject_ids[subset_key],
        num_classes=model_args.num_classes,
        allow_missing_samples=args.allow_missing_samples,
        include_metadata=True
    )
    assert ds.num_classes == model_args.num_classes
    for img, ys, meta in tqdm.tqdm(ds, desc=subset_key):
        with torch.no_grad():
            img = img.to(args.device)
            ys = ys.to(args.device)
            pred = model.predict(img[None, :, :, :])

            if model_args.num_classes in (4, 5):
                loss_dice = monai.losses.DiceLoss(
                    include_background=False,
                    softmax=True,
                    to_onehot_y=True,
                )(pred, ys[None, None, :, :])

            elif model_args.num_classes == 1:
                loss_dice = monai.losses.DiceLoss(
                    sigmoid=True,
                )(pred, ys[None, None, :, :])

            else:
                raise ValueError(f'unexpected num_classes: {self.num_classes}')

            if args.output is not None:
                with h5py.File(args.output, 'a') as h5_out:
                    group = h5_out.create_group(f'/scans/{meta["subject_id"]}/{meta["visit"]}/{meta["side"]}')
                    for k, v in meta.items():
                        group.attrs[k] = v
                    pred_g = group.create_dataset(
                        "segmentation_predicted",
                        data=pred[0].detach().cpu().numpy().astype('float16'),
                        **hdf5plugin.Blosc2(cname='blosclz', clevel=9,
                        filters=hdf5plugin.Blosc2.SHUFFLE))
                    pred_g.attrs["dice"] = loss_dice.item()
                    pred_g = group.create_dataset(
                        "segmentation_gt",
                        data=ys.detach().cpu().numpy().astype('uint8'),
                        **hdf5plugin.Blosc2(cname='blosclz', clevel=9,
                        filters=hdf5plugin.Blosc2.SHUFFLE))

            results.append(loss_dice)
