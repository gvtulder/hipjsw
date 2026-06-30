"""
Utility script to generate a contact sheet displaying a large
number of image crops and segmentations from an HDF5 file.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
import h5py
import hdf5plugin
import json
import pandas as pd
import skimage.color
import tqdm

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def compute_bbox(img):
    # https://stackoverflow.com/a/31402351
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_to_bbox(*img, mask):
    rmin, rmax, cmin, cmax = compute_bbox(mask)
    MARGIN = 20
    crop_size = max(rmax - rmin, cmax - cmin) // 2 + MARGIN
    rcenter = np.clip(
        (rmin + rmax) // 2,
        crop_size, img[0].shape[0] - crop_size
    )
    ccenter = np.clip(
        (cmin + cmax) // 2,
        crop_size, img[0].shape[1] - crop_size
    )
    crop = (
        slice(rcenter - crop_size,
              rcenter + crop_size),
        slice(ccenter - crop_size,
              ccenter + crop_size),
    )
    return [i[crop] for i in img]


parser = argparse.ArgumentParser()
parser.add_argument("--data", metavar="HDF5", required=True)
parser.add_argument("--predictions", metavar="HDF5", required=True)
parser.add_argument("--output", metavar="PDF", required=True)
parser.add_argument("--match", metavar="REGEX", default="")
parser.add_argument("--subset", metavar="N", type=int, default=10)
parser.add_argument("--joint-space-label", metavar="C", type=int, default=4)
parser.add_argument("--columns", metavar="N", type=int, default=2)
parser.add_argument("--crop-to-mask", action="store_true")
parser.add_argument("--sorted-samples", action="store_true")
args = parser.parse_args()


num_samples = args.subset
cols = args.columns
rows = args.subset // cols
NPLOTS = 5

# find all paths
possible_paths = []
with h5py.File(args.predictions, "r") as h5:
    for subject_id in h5["/scans"]:
        if re.match(args.match, subject_id):
            for visit in h5[f"/scans/{subject_id}"]:
                for side in h5[f"/scans/{subject_id}/{visit}"]:
                    if "segmentation_predicted" in h5[f"/scans/{subject_id}/{visit}/{side}"]:
                        possible_paths.append(f"/scans/{subject_id}/{visit}/{side}")

print(f"Found {len(possible_paths)} matches.")

# random shuffle
np.random.shuffle(possible_paths)
selected_samples = possible_paths[:num_samples]
if args.sorted_samples:
    selected_samples = sorted(selected_samples)

plt.figure(figsize=(cols * NPLOTS * 3, rows * 3))
with h5py.File(args.predictions, "r") as h5_seg:
    with h5py.File(args.data, "r") as h5_img:
        for i, sample_path in enumerate(tqdm.tqdm(selected_samples, desc="Plotting images")):
            img = h5_img[sample_path]["image"][:]
            seg_gt = h5_seg[sample_path].get("segmentation_gt")
            if seg_gt is not None:
                seg_gt = seg_gt[:]
            seg_pred = h5_seg[sample_path]["segmentation_predicted"]

            if seg_pred.ndim == 3 and seg_pred.shape[0] == 1:
                seg_pred = seg_pred[0]
                binary = True
            else:
                binary = False
                num_classes = seg_pred.shape[0]

            if binary and args.crop_to_mask:
                # crop to segmented region
                mask = np.logical_or(seg_gt > 0.5, seg_pred > 0)
                img, seg_gt, seg_pred = crop_to_bbox(
                    img, seg_gt, seg_pred, mask=mask)

            colormap = "gray"

            # original image
            plt.subplot(rows, cols * NPLOTS, i * NPLOTS + 1)
            img_clipped = np.clip(img, *np.percentile(img, [1, 99]))
            plt.imshow(img_clipped, cmap=colormap, interpolation="none")
            plt.title(sample_path.replace("/scans/", ""))
            plt.axis("off")

            if binary:
                # overlay
                plt.subplot(rows, cols * NPLOTS, i * NPLOTS + 2)
                bound_seg_pred = skimage.segmentation.find_boundaries(seg_pred > 0, mode="inner")
                bound_seg_gt = skimage.segmentation.find_boundaries(seg_gt > 0.5, mode="inner")
                overlay_rgb = np.repeat(img_clipped[:, :, None], repeats=3, axis=2)
                overlay_rgb -= overlay_rgb.min()
                overlay_rgb /= overlay_rgb.max()
                overlay_rgb[bound_seg_gt, :] = [0, 1, 1]
                overlay_rgb[bound_seg_pred, :] = [1, 0, 1]
                plt.imshow(overlay_rgb, interpolation="none")
                plt.title("Overlay")
                plt.axis("off")

                # predicted segmentation
                plt.subplot(rows, cols * NPLOTS, i * NPLOTS + 3)
                seg_pred_rgb = np.repeat(seg_pred[:, :, None], repeats=3, axis=2).astype(float)
                seg_pred_rgb = sigmoid(seg_pred_rgb)
                seg_pred_rgb -= seg_pred_rgb.min()
                seg_pred_rgb /= seg_pred_rgb.max()
                seg_pred_rgb[:, :, 1] = 0
                plt.imshow(seg_pred_rgb, interpolation="none")
                plt.title("Prediction")
                plt.axis("off")

                # ground truth segmentation
                plt.subplot(rows, cols * NPLOTS, i * NPLOTS + 4)
                seg_gt_rgb = np.repeat(seg_gt[:, :, None], repeats=3, axis=2).astype(float)
                seg_gt_rgb -= seg_gt_rgb.min()
                seg_gt_rgb /= seg_gt_rgb.max()
                seg_gt_rgb[:, :, 0] = 0
                plt.imshow(seg_gt_rgb, interpolation="none")
                plt.title("Ground truth")
                plt.axis("off")

                # combined segmentation
                plt.subplot(rows, cols * NPLOTS, i * NPLOTS + 5)
                combined_rgb = np.maximum(seg_pred_rgb, seg_gt_rgb)
                plt.imshow(combined_rgb, interpolation="none")
                plt.title("Combined")
                plt.axis("off")

            else:
                seg_pred_rgb = np.argmax(seg_pred, axis=0)

                # overlay
                plt.subplot(rows, cols * NPLOTS, i * NPLOTS + 2)
                plt.imshow(img_clipped, cmap=colormap, interpolation="none")
                plt.imshow(seg_pred_rgb, alpha=0.5, vmin=0, vmax=(1 if binary else num_classes), interpolation="none")
                plt.title("Overlay")
                plt.axis("off")

                # predicted segmentation
                plt.subplot(rows, cols * NPLOTS, i * NPLOTS + 3)
                plt.imshow(seg_pred_rgb, vmin=0, vmax=(1 if binary else num_classes), interpolation="none")
                plt.title("Prediction")
                plt.axis("off")

                # ground truth segmentation
                plt.subplot(rows, cols * NPLOTS, i * NPLOTS + 4)
                plt.imshow(seg_gt, vmin=0, vmax=(1 if binary else num_classes), interpolation="none")
                plt.title("Ground truth")
                plt.axis("off")

                # combined joint space segmentation
                plt.subplot(rows, cols * NPLOTS, i * NPLOTS + 5)
                combined_rgb = np.zeros((*seg_pred_rgb.shape, 3))
                combined_rgb[:, :, 0] = (seg_pred_rgb == args.joint_space_label)
                combined_rgb[:, :, 1] = (seg_gt == args.joint_space_label)
                combined_rgb[:, :, 2] = np.logical_or(seg_pred == args.joint_space_label, seg_gt == args.joint_space_label)
                plt.imshow(combined_rgb, interpolation="none")
                plt.title("Combined joint space")
                plt.axis("off")


plt.tight_layout()
plt.savefig(args.output)

