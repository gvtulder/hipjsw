"""
Prepare DICOM and JPEG images for YOLOLite training.

This converts DICOM and JPEG images to downsampled JPEG images,
and uses BoneFinder points files to compute the training ROIs
for left and right hips.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import argparse
import numpy as np
import os
import os.path
import h5py
import hdf5plugin
import imageio
import json
import pandas as pd
import re
import scipy.interpolate
import skimage
import tqdm
import sys
from multiprocessing import Pool

import bonefinder
import dicom_util


def load_single_image(subset, dataset, subject_id, visit, side,
                      input_dicom, input_points, input_manual,
                      target_pixel_spacing, forced_input_pixel_spacing=None,
                      min_aspect_ratio=None):
    # load image with metadata
    if input_dicom.lower().endswith('.dcm'):
        _, img_pixels, source_pixel_spacing = dicom_util.load_dicom_image(input_dicom)
    elif input_dicom.lower().endswith('.jpg'):
        img_pixels = imageio.v2.imread(input_dicom).astype(float)
        if img_pixels.ndim == 3:
            img_pixels = np.mean(img_pixels, axis=2)
        source_pixel_spacing = None

    # determine pixel spacing
    if source_pixel_spacing is None:
        assert forced_input_pixel_spacing is not None, \
               'input file does not contain pixel spacing, expected forced pixel spacing'
        input_pixel_spacing = [forced_input_pixel_spacing, forced_input_pixel_spacing]
    else:
        if forced_input_pixel_spacing is not None:
            matched_forced = \
                np.round(source_pixel_spacing[0], 3) == np.round(forced_input_pixel_spacing, 3) and \
                np.round(source_pixel_spacing[1], 3) == np.round(forced_input_pixel_spacing, 3)
#           assert matched_forced, \
#                  f'forced pixel spacing given, but input file already contains pixel spacing, {source_pixel_spacing} != {forced_input_pixel_spacing}'
            if not matched_forced:
                print(f'ignoring forced pixel spacing: input file already contains pixel spacing, {source_pixel_spacing} != {forced_input_pixel_spacing}')
                forced_input_pixel_spacing = None
        input_pixel_spacing = source_pixel_spacing

    ########################
    ## BONEFINDER ANNOTATIONS
    ########################
    # load points from BoneFinder
    # coordinates are defined in mm (if pixel spacing is available)
    if input_points.lower().endswith('_l.pts') or input_points.lower().endswith('_r.pts'):
        points = bonefinder.BonefinderPoints(
            input_points, [side],
            pixel_spacing=(input_pixel_spacing if source_pixel_spacing is None else None),
        )
    elif input_points.lower().endswith('_rasl.pts'):
        # left hip (right on the image) annotated on mirrored image
        assert side == 'left'
        points = bonefinder.BonefinderPoints(
            input_points, [side],
            img_pixels.shape[1] * input_pixel_spacing[1],
            pixel_spacing=(input_pixel_spacing if source_pixel_spacing is None else None),
        )
    else:
        points = bonefinder.BonefinderPoints(
            input_points,
            pixel_spacing=(input_pixel_spacing if source_pixel_spacing is None else None),
        )

    ########################
    ## RESAMPLING
    ########################
    # resample to the required resolution
    if target_pixel_spacing is not None:
        scale_factor = input_pixel_spacing[0] / target_pixel_spacing
        img_pixels = skimage.transform.rescale(img_pixels, scale_factor)
        pixel_spacing = [target_pixel_spacing, target_pixel_spacing]
    else:
        pixel_spacing = input_pixel_spacing

    ########################
    ## COMPUTE BBOX
    ########################
    # find bbox centered on the femoral head
    circles = points.circles_in_pixels(pixel_spacing)
    circle = circles[f'{side} femoral head']
    crop_size_in_pixels = int(2 * circle['r'])
    offset_y = np.clip(int(circle['yc']) - crop_size_in_pixels // 2,
                       0, img_pixels.shape[0] - crop_size_in_pixels)
    offset_x = np.clip(int(circle['xc']) - crop_size_in_pixels // 2,
                       0, img_pixels.shape[1] - crop_size_in_pixels)

    bbox = {
        'center_y': int(circle['yc']) / img_pixels.shape[0],
        'center_x': int(circle['xc']) / img_pixels.shape[1],
        'bbox_height': crop_size_in_pixels / img_pixels.shape[0],
        'bbox_width': crop_size_in_pixels / img_pixels.shape[1],
    }

    ########################
    ## CROP
    ########################
    # crop extremely long images (e.g., full-leg X-rays) by removing the bottom
    aspect_ratio = img_pixels.shape[1] / img_pixels.shape[0]
    if min_aspect_ratio is not None and min_aspect_ratio > aspect_ratio:
        img_pixels = img_pixels[:int(img_pixels.shape[1] / min_aspect_ratio), :]
        aspect_ratio = img_pixels.shape[1] / img_pixels.shape[0]

    ########################
    ## NORMALIZATION
    ########################
    # normalize intensities
    intensity_offset = img_pixels.min()
    intensity_slope = img_pixels.max() - intensity_offset

    img_pixels = (img_pixels - intensity_offset) / intensity_slope

    ########################
    ## BONEFINDER STATS
    ########################
    femoral_head_radius = circles[f'{side} femoral head']['r']
    sourcil_radius = circles[f'{side} sourcil']['r']

    return {
        "subset": subset,
        "dataset": dataset,
        "subject_id": subject_id,
        "visit": visit,
        "side": side,
        "input_dicom": input_dicom,
        "input_points": input_points,
        "input_manual": input_manual,
        "target_pixel_spacing": target_pixel_spacing,
        "crop_size_in_pixels": crop_size_in_pixels,
        "img_pixels": img_pixels,
        "pixel_spacing": pixel_spacing,
        "source_pixel_spacing": source_pixel_spacing,
        "forced_input_pixel_spacing": forced_input_pixel_spacing,
        "input_pixel_spacing": input_pixel_spacing,
        "intensity_offset": intensity_offset,
        "intensity_slope": intensity_slope,
        "offset_y": offset_y,
        "offset_x": offset_x,
        "femoral_head_radius": femoral_head_radius,
        "sourcil_radius": sourcil_radius,
        "bbox": bbox,
        "femoral_head_circle": circle,
    }


def write_single_image(output_path, subset,
        dataset, subject_id, visit, side, input_dicom, input_points, input_manual,
        target_pixel_spacing, crop_size_in_pixels,
        img_pixels,
        pixel_spacing, source_pixel_spacing,
        forced_input_pixel_spacing, input_pixel_spacing,
        intensity_offset, intensity_slope, offset_y, offset_x,
        femoral_head_radius, sourcil_radius, bbox, femoral_head_circle):
    # write to output
    os.makedirs(f'{output_path}/images/{subset}', exist_ok=True)
    os.makedirs(f'{output_path}/labels/{subset}', exist_ok=True)
    os.makedirs(f'{output_path}/stats/{subset}', exist_ok=True)
    scan_id = os.path.basename(input_dicom)
    img_pixels = (np.clip(img_pixels, 0, 1) * 255).astype('uint8')
    imageio.imwrite(f'{output_path}/images/{subset}/{scan_id}.jpg', img_pixels)

    with open(f'{output_path}/labels/{subset}/{scan_id}.txt', 'a') as f:
        # class_id (left: 0, right: 1) center_x center_y bbox_width bbox_height
        class_id = 0 if side == 'left' else 1
        bbox_str = ' '.join([f'{bbox[k]}' for k in ['center_x', 'center_y', 'bbox_width', 'bbox_height']])
        f.write(f'{class_id} {bbox_str}\n')

    with open(f'{output_path}/stats/{subset}/{scan_id}.{side}.json', 'w') as f:
        f.write(json.dumps({
            'input_pixel_spacing': [float(v) for v in input_pixel_spacing],
            'pixel_spacing': [float(v) for v in pixel_spacing],
            'femoral_head_radius': float(femoral_head_radius),
            'femoral_head_circle': femoral_head_circle,
        }))


parser = argparse.ArgumentParser()
parser.add_argument('--input-csv', metavar='CSV', nargs='+', required=True,
                    help='list of subjects and images as CSV')
parser.add_argument('--subsets', metavar='CSV', required=True,
                    help='list with subset for each list')
parser.add_argument('--images-path', metavar='PATH', required=True,
                    help='base directory for the images')
parser.add_argument('--points-path', metavar='PATH', required=True,
                    help='base directory for the point files')
parser.add_argument('--output-path', metavar='PATH', required=True,
                    help='output directory')
parser.add_argument('--visit', metavar='VISIT', default='T00',
                    help='default visit if not given in CSV')
parser.add_argument('--target-pixel-spacing', metavar='SPACING', type=float,
                    help='resample image to target spacing (mm/pixel)')
parser.add_argument('--min-aspect-ratio', metavar='ASPECT', type=float,
                    help='crop the image (remove bottom part) to this minimum aspect ratio')
parser.add_argument('--skip-missing-files', action='store_true',
                    help='ignore file-not-found errors')
parser.add_argument('--debug-subset', action='store_true',
                    help='only process a few subjects for debugging')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--num-workers', metavar='N', type=int, default=1,
                    help='load images in parallel')
args = parser.parse_args()


# load subsets
subject_subsets = {}
with open(args.subsets, 'r') as f:
    for line in f:
        if line.strip() != '':
            subject, subset = line.strip().split(',')
            subject_subsets[subject] = subset


# list all images to be converted
all_files = []
for csv_filename in args.input_csv:
    df = pd.read_csv(csv_filename)
    df = df.rename(columns=lambda s: s.lower())
    if 'exclude' in df:
        # skip excluded rows
        df = df[df['exclude'] != 1]
    for row in df.to_dict(orient='records'):
        subject = row.get('subject_id') or row.get('coach_id')
        subset = subject_subsets.get(subject)
        if subset is None:
            continue
        input_dicom = os.path.join(row.get('linux_path_img') or args.images_path,
                                   row.get('dicom') or row.get('img_names_baseline') or row.get('img_names'))
        input_points = os.path.join(row.get('linux_path_pts') or args.points_path,
                                   row.get('points') or row.get('pts_names_baseline') or row.get('pts_names'))
        if row.get('manual') or row.get('manual_names_baseline') or row.get('manual_names'):
            input_manual = os.path.join(row.get('linux_path_manual') or args.manual_path,
                                        row.get('manual') or row.get('manual_names_baseline') or row.get('manual_names'))
        else:
            input_manual = None
        if not args.skip_missing_files:
            assert os.path.exists(input_dicom), f'File not found: {input_dicom}'
            assert os.path.exists(input_points), f'File not found: {input_points}'
        if os.path.exists(input_dicom) and os.path.exists(input_points):
            forced_input_pixel_spacing = row.get('img_spacing')
            if forced_input_pixel_spacing in (0, 0.0, "", "0", "0.0"):
                forced_input_pixel_spacing = None
            if forced_input_pixel_spacing is not None:
                forced_input_pixel_spacing = float(forced_input_pixel_spacing)
            if forced_input_pixel_spacing and np.isnan(forced_input_pixel_spacing):
                forced_input_pixel_spacing = None
            all_files.append({
                'subset': subset,
                'dataset': row.get('dataset') or row.get('cohort_name'),
                'subject_id': row.get('subject_id') or row.get('coach_id'),
                'visit': row.get('visit', args.visit),
                'side': row.get('side') or row.get('measurement_side'),
                'input_dicom': input_dicom,
                'input_points': input_points,
                'input_manual': input_manual,
                'forced_input_pixel_spacing': forced_input_pixel_spacing,
                'min_aspect_ratio': args.min_aspect_ratio,
            })
        else:
            print(f'Skipped row with missing file: {input_dicom} or {input_points}')

if args.debug_subset:
    all_files = all_files[:30]


def do_load_single_image(task):
    try:
        return load_single_image(
                **task,
                target_pixel_spacing=args.target_pixel_spacing)
    except Exception as e:
        # skip this image for now
        print(f"FAILED TO CONVERT {task['input_dicom']}: ")
        print(task)
        print(e)
        import traceback
        traceback.print_exc()
        print()
        return None


# pool for loading images in parallel
with Pool(args.num_workers) as pool:
    for image_data in tqdm.tqdm(pool.imap_unordered(do_load_single_image, all_files),
                                total=len(all_files),
                                disable=args.quiet, desc='Converting'):
        if image_data is not None:
            try:
                write_single_image(args.output_path, **image_data)
            except Exception as e:
                print(f"FAILED TO WRITE {image_data['subject_id']}/{image_data['visit']}/{image_data['side']}")
                print(e)
                print()
