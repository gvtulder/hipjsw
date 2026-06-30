"""
Script to train a segmentation model.
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

import torch
import torch.nn
import lightning.pytorch as pl
import monai

from dataset import datasets
from model import models

# for the TU Delft DAIC cluster
# import patch_fsspec

# on the cluster: limit the number of CPU threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# load experiment settings
parser = argparse.ArgumentParser()

# - data
parser_data = parser.add_argument_group('data')
parser_data.add_argument('--model', metavar='MODEL', required=True,
                         choices=list(models),
                         help='model for task')
parser_data.add_argument('--dataset', metavar='CLASS', required=True,
                         choices=list(datasets),
                         help='dataset for task')
parser_data.add_argument('--data', metavar='HDF5', required=True, nargs='+',
                         help='dataset file (HDF5)')
parser_data.add_argument('--data-split', metavar='CSV', required=True, nargs='+',
                         help='list of subject IDs with subsets')
parser_data.add_argument('--allow-missing-samples', action='store_true',
                         help='do not raise errors on missing samples')
parser_data.add_argument('--data-augment', metavar='AUGMENT',
                         choices=('flip', 'rotate', 'elastic', 'elastic-strong',
                                  'intensity3', 'intensity4', 'gamma',
                                  'jsw'),
                         nargs='+', default=[],
                         help='apply data augmentations')
parser_data.add_argument('--data-debug-subset', action='store_true',
                         help='use only a small subset of the dataset for debugging')
parser_data.add_argument('--dataset-train', metavar='CLASS',
                         choices=list(datasets),
                         help='dataset for task (training data only)')
parser_data.add_argument('--num-classes', metavar='NUM', type=int, default=5,
                         help='the number of classes')

# - hyperparameters
parser_hyp = parser.add_argument_group('hyperparameters')
parser_hyp.add_argument('--lr', metavar='LR', default=1e-3, type=float)
parser_hyp.add_argument('--weight-decay', metavar='WD', type=float)
parser_hyp.add_argument('--mb-size', metavar='N', default=16, type=int)
parser_hyp.add_argument('--num-workers', metavar='N', default=0, type=int)
parser_hyp.add_argument('--max-epochs', metavar='N', default=100, type=int)

# - experiment
parser_exp = parser.add_argument_group('experiment')
parser_exp.add_argument('--seed', metavar='SEED', default=123, type=int,
                        help='random seed for the experiments')
parser_exp.add_argument('--experiment-name', metavar='NAME',
                        help='experiment name for log')
parser_exp.add_argument('--log-dir', metavar='LOGS', default='logs/',
                        help='tensorboard log directory')
parser_exp.add_argument('--checkpoint-dir', metavar='CHECKPOINTS', default='checkpoints/',
                        help='checkpoint directory')
parser_exp.add_argument('--resume-checkpoint', metavar='CHECKPOINT',
                        help='resume from checkpoint')
parser_exp.add_argument('--pretrained-checkpoint', metavar='CHECKPOINT',
                        help='pretrained weights from checkpoint')

args = parser.parse_args()
vargs = vars(args)


# set numpy, torch, python.random seeds
pl.seed_everything(args.seed, workers=True)


# model
print('Preparing model')
model_args = {}
if args.weight_decay is not None:
    model_args['weight_decay'] = args.weight_decay
model = models[args.model](lr=args.lr, num_classes=args.num_classes, **model_args)

# load the train/validation split
subject_ids = {}
for filename in args.data_split:
    with open(filename, 'r') as f:
        for line in f:
            subject, subset_key = line.strip().split(',')
            if subset_key not in subject_ids:
                subject_ids[subset_key] = []
            subject_ids[subset_key].append(subject)

# construct the datasets and dataloaders
ds = {}
dataloaders = {}
print('Loading datasets:')
for subset_key in ['train', 'val']:
    if args.data_debug_subset:
        subject_ids[subset_key] = subject_ids[subset_key][:10]

    if 'train' in subset_key:
        augmentation = args.data_augment
    else:
        augmentation = []

    # override dataset for training?
    dataset_cls = args.dataset
    if 'train' in subset_key and args.dataset_train:
        dataset_cls = args.dataset_train

    ds[subset_key] = datasets[dataset_cls](
        filenames=args.data,
        subjects=subject_ids[subset_key],
        augment=augmentation,
        num_classes=args.num_classes,
        allow_missing_samples=args.allow_missing_samples,
    )
    print(f'- {subset_key} subjects: {len(subject_ids[subset_key])}')
    print(f'- {subset_key} samples: {len(ds[subset_key])}')

    assert ds[subset_key].num_classes == args.num_classes

    if hasattr(ds[subset_key], 'sample_weights') and 'train' in subset_key:
        sampler = torch.utils.data.WeightedRandomSampler(
            ds[subset_key].sample_weights,
            num_samples=len(ds[subset_key]),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = ('train' in subset_key)

    dataloaders[subset_key] = torch.utils.data.DataLoader(
        ds[subset_key],
        batch_size=args.mb_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
    )


# loggers
loggers = []
if args.log_dir:
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.log_dir)
    loggers.append(tb_logger)


# checkpoints
if args.checkpoint_dir:
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='val/loss',
            filename='best-val-loss-{epoch}-{step}',
            dirpath=args.checkpoint_dir,
            save_top_k=2,
            save_last=True,
        ),
    #   pl.callbacks.ModelCheckpoint(
    #       filename='{epoch}-{step}',
    #       dirpath=args.checkpoint_dir,
    #       every_n_epochs=5,
    #       save_last=True,
    #       save_top_k=-1,
    #   ),
    ]
    with open(f'{args.checkpoint_dir}/args.json', 'w') as f:
        json.dump(vargs, f)
else:
    checkpoint_callbacks = []


if args.pretrained_checkpoint is not None:
    print(f'Load weights from checkpoint {args.pretrained_checkpoint}')
    checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])


if args.resume_checkpoint is not None:
    print(f'Resume from checkpoint {args.resume_checkpoint}')


# train
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    logger=loggers,
    enable_progress_bar=False,
    callbacks=checkpoint_callbacks,
)
trainer.fit(
    model=model,
    train_dataloaders=dataloaders['train'],
    val_dataloaders=dataloaders['val'],
    ckpt_path=args.resume_checkpoint,
)


print(args)


