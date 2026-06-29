"""
Utility functions for 2D measurements and contours.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import numpy as np
import skimage
import scipy.stats

def sigmoid(x):
    """Apply the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))

def compute_bbox(mask):
    """Compute the bounding box from a mask."""
    # https://stackoverflow.com/a/31402351
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def compute_bbox_with_margin(mask, margin=20, square=True):
    """Compute the bounding box from a mask, adding a number of pixels as a margin.

    Parameters
    ----------
    mask : numpy array
        a mask (True or > 0)
    margin : int
        a margin in pixels to add on each side
    square : bool
        if True, returns a square crop with a centered object

    """
    rmin, rmax, cmin, cmax = compute_bbox(mask)
    crop_size_r = (rmax - rmin) // 2 + margin
    crop_size_c = (cmax - cmin) // 2 + margin
    if square:
        crop_size_r = crop_size_c = max(crop_size_r, crop_size_c)
    rcenter = np.clip(
        (rmin + rmax) // 2,
        crop_size_r, mask.shape[0] - crop_size_r
    )
    ccenter = np.clip(
        (cmin + cmax) // 2,
        crop_size_c, mask.shape[1] - crop_size_c
    )
    return rcenter - crop_size_r, \
           rcenter + crop_size_r, \
           ccenter - crop_size_c, \
           ccenter + crop_size_c

def crop_to_bbox(mask, margin=20, square=True):
    """Crop the image to a bounding box."""
    rmin, rmax, cmin, cmax = compute_bbox(mask, margin, square)
    return slice(rmin, rmax), slice(cmin, cmax)

def select_largest_object(mask):
    """Return a binary mask with only the largest object."""
    labels, num = skimage.measure.label(mask, return_num=True)
    areas = [np.sum(labels == l) for l in range(num + 1)]
    largest_object = np.argmax(areas[1:]) + 1
    return labels == largest_object

def mean_curve_distance(c1, c2):
    """Compute the mean distance between two curves.

    For each point in ``c1``, compute the distance to the closest
    point in ``c2``, then compute the mean.
    """
    return np.mean(np.min(np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=2), axis=1))

def pointwise_distance_to_curve(points, curve):
    """Compute the mean distance between points on two curves.

    For each point in ``c1``, compute the distance to the closest
    point in ``c2``.
    """
    return np.min(np.linalg.norm(points[:, None, :] - curve[None, :, :], axis=2), axis=1)

def find_circ_start_end(x):
    """"Find the starting point of a series of True elements."""
    # print(find_circ_start_end(np.array([False, False, True, True, True, True, False])))
    # print(find_circ_start_end(np.array([True, True, False, False, False, True, True])))
    idxs = np.where(x)[0]
    center = int(scipy.stats.circmean(idxs, high=x.shape[0]))
    shift = (center + x.shape[0] // 2) % x.shape[0]
    idxs = (idxs - shift) % x.shape[0]
    return int(min(idxs) + shift) % x.shape[0], \
           int(max(idxs) + shift + 1) % x.shape[0]

def crop_contour(x, start, end):
    """Crop the list to the given start and end position, rolling the list if necessary."""
    return np.roll(x, -start, axis=0)[0:((end - start + x.shape[0] - 1) % x.shape[0] + 1)]

def polar_coordinates(contour, origin):
    """Convert the contour x,y coordinates to polar coordinates."""
    contour_wrt_center = contour - origin
    angle = np.arctan2(contour_wrt_center[:, 1], contour_wrt_center[:, 0])
    radius = np.linalg.norm(contour_wrt_center, axis=1)
    return angle, radius
