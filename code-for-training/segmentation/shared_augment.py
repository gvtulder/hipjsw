"""
Class with shared data augmentation functions.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import numpy as np
import skimage.transform
import torch
import elasticdeform
import augment_jsw


class SharedAugmentations:
    def _augment(self, img, seg, image_attrs={}):
        assert all(a in ('rotate', 'elastic', 'elastic-strong',
                         'intensity3', 'intensity4', 'gamma',
                         'jsw') for a in self.augment), \
               f'invalid augmentation in {self.augment}'

        if 'rotate' in self.augment:
            img, seg = self._augment_rotate(img, seg)

        if 'elastic' in self.augment:
            assert 'rotate' not in self.augment
            img, seg = self._augment_elastic(img, seg)

        if 'elastic-strong' in self.augment:
            assert 'rotate' not in self.augment
            img, seg = self._augment_elastic_strong(img, seg)

        if 'jsw' in self.augment:
            assert 'rotate' not in self.augment
            img, seg = self._augment_jsw(img, seg, image_attrs)

        if 'intensity3' in self.augment:
            img, seg = self._augment_intensity3(img, seg)

        if 'intensity4' in self.augment:
            img, seg = self._augment_intensity4(img, seg)

        if 'gamma' in self.augment:
            img, seg = self._augment_gamma(img, seg)

        return img, seg

    def _augment_rotate(self, img, seg):
        # random rotate by -10 to 10 degrees around the center of the image
        angle = torch.randint(low=-10, high=10, size=(1,)).item()
        orig_center = [img.shape[1] / 2 - 0.5, img.shape[0] / 2 - 0.5]
        orig_shape = img.shape
        img = skimage.transform.rotate(img, angle=angle, center=orig_center, mode='constant', cval=np.min(img))
        seg = skimage.transform.rotate(seg, angle=angle, center=orig_center, mode='constant', cval=np.min(seg), order=0)
        return img, seg

    def _augment_elastic(self, img, seg):
        # random rotate by -10 to 10 degrees around the center of the image
        angle = np.random.uniform(-10, 10)
        # random scale between 0.8 and 1.2
        scale = np.random.uniform(0.9, 1.1)
        img, seg = elasticdeform.deform_random_grid(
            [img, seg], sigma=5, points=3, rotate=angle, zoom=scale, order=[3, 0])
        return img, seg

    def _augment_elastic_strong(self, img, seg):
        # random rotate by -15 to 15 degrees around the center of the image
        angle = np.random.uniform(-15, 15)
        # random scale between 0.8 and 1.3
        scale = np.random.uniform(0.8, 1.3)
        img, seg = elasticdeform.deform_random_grid(
            [img, seg], sigma=10, points=7, rotate=angle, zoom=scale, order=[3, 0])
        return img, seg

    def _augment_jsw(self, img, seg, image_attrs):
        femoral_head_radius = image_attrs['femoral_head_radius']
        if femoral_head_radius is not None:
            strength = np.random.uniform(0, 40)
            img, seg = augment_jsw.augment_jsw_displacement(
                img, seg,
                femoral_head_radius=femoral_head_radius,
                strength=strength
            )
        return img, seg

    def _augment_intensity3(self, img, seg):
        # intensity
        def add_random_scale(x):
            ycoord = np.linspace(0, 1, x.shape[0])[:, None]
            xcoord = np.linspace(0, 1, x.shape[1])[None, :]
            scale = np.random.normal(0, 0.1)
            xfreq = np.random.normal(0.5, 1.0)
            yfreq = np.random.normal(0.5, 1.0)
            phase = np.random.randint(0, 1)
            return x * (1 + scale * np.sin((xfreq * xcoord + yfreq * ycoord + phase) * 2 * np.pi))

        scale = np.ones_like(img)
        scale = add_random_scale(scale)
        scale = add_random_scale(scale)
        scale = add_random_scale(scale)
        scale = np.clip(scale, 0.7, 1.3)
        return img * scale, seg

    def _augment_intensity4(self, img, seg):
        # intensity
        def add_random_scale(x):
            ycoord = np.linspace(0, 1, x.shape[0])[:, None]
            xcoord = np.linspace(0, 1, x.shape[1])[None, :]
            scale = np.random.normal(0, 0.3)
            xfreq = np.random.normal(0.5, 1.0)
            yfreq = np.random.normal(0.5, 1.0)
            phase = np.random.randint(0, 1)
            return x * (1 + scale * np.sin((xfreq * xcoord + yfreq * ycoord + phase) * 2 * np.pi))

        scale = np.ones_like(img)
        scale = add_random_scale(scale)
        scale = add_random_scale(scale)
        scale = add_random_scale(scale)
        scale = add_random_scale(scale)
        scale = np.clip(scale, 0.5, 1.5)
        return img * scale, seg

    def _augment_gamma(self, img, seg):
        # intensity
        gamma = np.random.uniform(0.1, 1.2)
        shift = np.random.uniform(-0.2, 0.2)
        img_min, img_max = img.min(), img.max()
        img_aug = ((img - img_min) / img_max) ** gamma * img_max + img_min
        img_aug += (img_max - img_min) * shift
        return img_aug, seg
