"""
Some utility functions to augment a segmentation by deforming
the region around the femoral head contour.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import numpy as np
import skimage.filters


# compute displacement maps using Bézier curve on a circle
def displace_circular(shape, radius, x1=40, x2=40, displacement_strength=1, displacement_angle=0.4 * np.pi):
    img = np.ones(shape)
    cx, cy = shape[0] // 2, shape[1] // 2
    displacement_falloff = 1
    
    xs = np.arange(img.shape[1]) - cx
    ys = np.arange(img.shape[0]) - cy
    
    angle_xy = np.arctan2(ys[:, None], xs[None, :])
    radius_xy = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
    radius_offset_xy = radius_xy - radius

    # compute t given x in a quadratic Bézier curve
    def t(x, x0, x1, x2):
        return (np.sqrt(np.maximum(0,
            -x0 * x2 + x0 * x +
            x1 ** 2 - 2 * x1 * x +
            x2 * x)) + x0 - x1) \
        / (x0 - 2 * x1 + x2)

    # x0, x1, x2 = 0, 40, 40
    assert x1 <= x2
    assert 2 * x1 != x2
    x0 = 0
    ts = np.clip(t(radius_offset_xy, x0=x0, x1=x1, x2=x2), 0, 1)
    ts_mirror = np.clip(t(-radius_offset_xy, x0=x0, x1=x1, x2=x2), 0, 1)
    y0, y1, y2 = 0, x1, 0
    radius_diff_xy = \
        y1 + (1 - ts) ** 2 * (y0 - y1) + ts ** 2 * (y2 - y1) + \
        -y1 + (1 - ts_mirror) ** 2 * (y1 - y0) + ts_mirror ** 2 * (y1 - y2)

    displacement_strength = (np.cos(angle_xy + displacement_angle) * displacement_falloff + 1) * displacement_strength
    displacement_x = displacement_strength * radius_diff_xy * np.cos(angle_xy)
    displacement_y = displacement_strength * radius_diff_xy * np.sin(angle_xy)

    return displacement_y, displacement_x

def augment_jsw_displacement(img, seg, femoral_head_radius, strength=40):
    # strength in pixels, can be made random, [0, 40] works well
    displacement_y, displacement_x = displace_circular(img.shape,
                                                       radius=femoral_head_radius,
                                                       x1=strength, x2=strength,
                                                       displacement_strength=1,
                                                       displacement_angle=0.5 * np.pi)

    displacement_x = skimage.filters.gaussian(displacement_x, sigma=15)
    displacement_y = skimage.filters.gaussian(displacement_y, sigma=15)

    inverse_map = np.array([
        np.arange(img.shape[0])[:, None] + displacement_y,
        np.arange(img.shape[1])[None, :] + displacement_x,
    ])
    img_warp = skimage.transform.warp(img, inverse_map, order=3)
    seg_warp = skimage.transform.warp(seg, inverse_map, order=0)

    return img_warp, seg_warp


if __name__ == '__main__':
    import h5py
    import hdf5plugin
    h5 = h5py.File('only-EVAL-for-hip-segmentation-v5-20260126-0.2mm-512x512.h5', 'r')
    subject = h5['/scans/CHECK-07005/T00/right']
    img = subject['image'][:]
    seg = subject['segmentation'][:]
    femoral_head_radius = subject['image'].attrs['femoral_head_radius']
    sourcil_radius = subject['image'].attrs['sourcil_radius']
    augment_jsw_displacement(img, seg, femoral_head_radius, 30)
    
