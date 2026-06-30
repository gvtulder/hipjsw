"""
Data loading classes to load training data from HDF5 files.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import h5py
import hdf5plugin
import numpy as np
import skimage.transform
import re
import torch
import torch.utils.data
import pandas as pd

import shared_augment

datasets = {}
def register_dataset(cls):
    datasets[cls.__name__] = cls
    return cls


@register_dataset
class SegmentationDataset(torch.utils.data.Dataset, shared_augment.SharedAugmentations):
    def __init__(self, filenames, subjects, include_metadata=False,
                 augment=[], img_dtype=torch.float32, seg_dtype=torch.long, num_classes=5,
                 allow_missing_samples=False):
        super().__init__()
        self.filenames = filenames
        self.subjects = subjects
        self.include_metadata = include_metadata
        self._load_samples(subjects, allow_missing_samples)

        self.augment = augment
        self.img_dtype = img_dtype
        self.seg_dtype = seg_dtype

        self.num_classes = num_classes

    def __getitem__(self, index):
        filename, subject_id, visit, side = self.samples[index]
        with h5py.File(filename, 'r') as h5:
            # find subject, visit, image, side
            group = h5[f'/scans/{subject_id}/{visit}/{side}']
            image = group['image']
            image_attrs = dict(image.attrs)

            # load image data
            img = image[:]

            # load segmentation data
            seg = group['segmentation'][:]

            img, seg = self._augment(img, seg, image_attrs)

            # correct output format and dtype
            img = torch.tensor(img[None, :, :], dtype=self.img_dtype)
            seg = torch.tensor(seg, dtype=self.seg_dtype)

            # collect metadata
            if self.include_metadata:
                meta = {}
                meta.update(group.attrs)
                meta.update(image_attrs)

        out = [img, seg]
        if self.include_metadata:
            out = [*out, meta]
        return out

    def _load_samples(self, subjects, allow_missing_samples):
        self.samples = []
        subjects_seen = {}
        for filename in self.filenames:
            with h5py.File(filename, 'r') as h5:
                for subject in subjects:
                    # does the subject include a visit or side?
                    if '/' in subject:
                        subject_id, *subject_visit = subject.split('/')
                        subject_side = subject_visit[1] if len(subject_visit) > 1 else None
                        subject_visit = subject_visit[0]
                    else:
                        subject_id = subject
                        subject_side = None
                        subject_visit = None
                    if f'/scans/{subject_id}' not in h5 \
                        and (len(self.filenames) > 1 or allow_missing_samples):
                        # maybe in the next file?
                        continue
                    subjects_seen[subject] = filename
                    for visit, visit_g in h5[f'/scans/{subject_id}'].items():
                        if subject_visit and subject_visit != visit:
                            continue
                        for side, side_g in visit_g.items():
                            if subject_side and subject_side != side:
                                continue
                            sample = (filename, subject_id, visit, side)
                            sample = self._filter_sample(sample)
                            if sample:
                                self.samples.append(sample)
        if not allow_missing_samples:
            assert len(subjects_seen) == len(subjects), \
                f'not all subjects found? {len(subjects_seen)} != {len(subjects)}'

    def _filter_sample(self, sample):
        return sample

    def __len__(self):
        return len(self.samples)


# binary segmentation task with joint space vs background
@register_dataset
class SegmentationDataset_JointSpaceOnly(SegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 1

    def __getitem__(self, index):
        img, seg, *meta = super().__getitem__(index)
        # joint space has label 4
        seg = (seg == 4).to(self.seg_dtype)
        return img, seg, *meta


if __name__ == '__main__':
    # demo code

    ds = SegmentationDataset(
        filenames=["data/all-for-hip-prediction-20250429-0.3mm-512x512.h5"],
        subjects=["CHECK-01102", "CHECK-05045"],
    )
    print(ds)
    img, seg = ds[0]
    print(img, seg)
