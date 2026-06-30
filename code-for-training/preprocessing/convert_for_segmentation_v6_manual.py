"""
Prepare DICOM and JPEG images for segmentation model training.

This loads DICOM and JPEG image, normalizes, crops to the hip ROI
centered around the femoral head, and writes the croppd images to
an HDF5 file that can be read by the training scripts.

This uses the manual annotations (in JSON format) to determine the
segmentation outlines.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import argparse
import numpy as np
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

# prepare HDF5 files for a joint space segmentation task
# based on a CSV list of input files

LABELS = {
    'ignore': 0,
    'background': 1,
    'femur': 2,
    'joint space': 3,
    'sourcil': 4,
}

def load_single_image(dataset, subject_id, visit, side, input_dicom, input_points, input_manual,
        target_pixel_spacing, crop_size_in_pixels, forced_input_pixel_spacing=None):
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
    ## MANUAL ANNOTATION CURVES
    ########################
    if input_manual:
        with open(input_manual, 'r') as f:
            doc = json.load(f)
        manual_curves = {}
        # curves in pixels
        manual_curves['sourcil'] = np.array(doc[f'smooth {side} sourcil'])
        manual_curves['femur'] = np.array(doc[f'smooth {side} femur'])
        # make sure to start curves on the left
        if manual_curves['sourcil'][0, 0] > manual_curves['sourcil'][-1, 0]:
            manual_curves['sourcil'][::-1, :]
        if manual_curves['femur'][0, 0] > manual_curves['femur'][-1, 0]:
            manual_curves['femur'][::-1, :]
        # compute curves as mm
        manual_curves = {
            f'{side} sourcil': manual_curves['sourcil'] * input_pixel_spacing[0],
            f'{side} femur': manual_curves['femur'] * input_pixel_spacing[0],
        }
    else:
        manual_curves = None

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
    ## SEGMENTATION
    ########################
    # draw segmentation mask
    segmentation = np.zeros(shape=img_pixels.shape, dtype=np.uint8)

    manual_curves = {
        key: manual_curves[key] / pixel_spacing
        for key in manual_curves.keys()
    }
    curves = points.curves_in_pixels(pixel_spacing)
    circles = points.circles_in_pixels(pixel_spacing)

    # smoothing
    def make_smooth_contour(contour, smoothness=0.1, num=None):
        if num is None:
            num = contour.shape[0]
        spl, u = scipy.interpolate.make_splprep(
            contour.T, s=contour.shape[0] * smoothness, k=3)
        uu1 = np.linspace(u[0], u[-1], num)
        return np.array(spl(uu1)).T

    smooth_manual_curves = {
        key: make_smooth_contour(manual_curves[key], smoothness=1, num=100)
        for key in manual_curves.keys()
    }

    # find lateral end of femur
    if (smooth_manual_curves[f'{side} femur'][0, 1] <
        smooth_manual_curves[f'{side} femur'][-1, 1]):
        lateral_end_femur = 0
        medial_end_femur = smooth_manual_curves[f'{side} femur'].shape[0] - 1
    else:
        lateral_end_femur = smooth_manual_curves[f'{side} femur'].shape[0] - 1
        medial_end_femur = 0

    regions = [
        ('background', np.array([
            # above lateral end of femur
            [smooth_manual_curves[f'{side} femur'][lateral_end_femur, 0], 0],
            # lateral end of femur
            smooth_manual_curves[f'{side} femur'][lateral_end_femur, :],
            # medial end of femur
            smooth_manual_curves[f'{side} femur'][medial_end_femur, :],
            # right of medial end of femur
            [(0 if side == "left" else img_pixels.shape[1]),
             smooth_manual_curves[f'{side} femur'][medial_end_femur, 1]],
            # top right
            [(0 if side == "left" else img_pixels.shape[1]), 0],
        ])),
        ('sourcil', np.array([
            # 2 mm above the sourcil
            *smooth_manual_curves[f'{side} sourcil'],
            *(smooth_manual_curves[f'{side} sourcil'][::-1, :]
              - ([0, 2] / np.array(pixel_spacing))),
        ])),
        ('joint space', np.array([
            # sourcil
            *smooth_manual_curves[f'{side} sourcil'],
            # to center of femoral head
            [circles[f'{side} femoral head']['xc'],
             circles[f'{side} femoral head']['yc']],
        ])),
        ('femur', np.array([
            # proximal femur
            *smooth_manual_curves[f'{side} femur'],
        ])),
    ]

    # add regions to mask
    for name, region in regions:
        mask = skimage.draw.polygon2mask(
            segmentation.shape,
            region[:, [1, 0]]
        )
        segmentation[mask] = LABELS[name]

    ########################
    ## CROPPING / FLIPPING
    ########################
    # crop the hip centered on the femoral head
    circle = circles[f'{side} femoral head']
    offset_y = np.clip(int(circle['yc']) - crop_size_in_pixels // 2,
                       0, img_pixels.shape[0] - crop_size_in_pixels)
    offset_x = np.clip(int(circle['xc']) - crop_size_in_pixels // 2,
                       0, img_pixels.shape[1] - crop_size_in_pixels)
    img_pixels_crop = img_pixels[
        offset_y:offset_y + crop_size_in_pixels,
        offset_x:offset_x + crop_size_in_pixels
    ]
    segmentation_crop = segmentation[
        offset_y:offset_y + crop_size_in_pixels,
        offset_x:offset_x + crop_size_in_pixels
    ]

    # check cropped size
    assert segmentation_crop.shape[0] == crop_size_in_pixels, \
        f'incorrect image size after cropping {segmentation_crop.shape}'
    assert segmentation_crop.shape[1] == crop_size_in_pixels, \
        f'incorrect image size after cropping {segmentation_crop.shape}'

    # flip left to right
    if side == 'left':
        img_pixels_crop = img_pixels_crop[:, ::-1]
        segmentation_crop = segmentation_crop[:, ::-1]

    ########################
    ## NORMALIZATION
    ########################
    # normalize intensities
    img_pixels_crop = img_pixels_crop.astype(float)
    percentile = np.percentile(img_pixels_crop.flatten(), [5, 95])
    intensity_offset = percentile[0]
    intensity_slope = percentile[1] - percentile[0]
    img_pixels_crop = (img_pixels_crop - intensity_offset) / intensity_slope

    ########################
    ## BONEFINDER STATS
    ########################
    femoral_head_radius = circles[f'{side} femoral head']['r']
    sourcil_radius = circles[f'{side} sourcil']['r']

    return {
        "dataset": dataset,
        "subject_id": subject_id,
        "visit": visit,
        "side": side,
        "input_dicom": input_dicom,
        "input_points": input_points,
        "input_manual": input_manual,
        "target_pixel_spacing": target_pixel_spacing,
        "crop_size_in_pixels": crop_size_in_pixels,
        "img_pixels_crop": img_pixels_crop,
        "segmentation_crop": segmentation_crop,
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
    }


def write_single_image(h5, dataset, subject_id, visit, side, input_dicom, input_points, input_manual,
        target_pixel_spacing, crop_size_in_pixels,
        img_pixels_crop, segmentation_crop,
        pixel_spacing, source_pixel_spacing,
        forced_input_pixel_spacing, input_pixel_spacing,
        intensity_offset, intensity_slope, offset_y, offset_x,
        femoral_head_radius, sourcil_radius):
    # write to hdf5
    group = h5.require_group(f'/scans/{subject_id}/{visit}/{side}')
    group.attrs['dataset'] = dataset
    group.attrs['subject_id'] = subject_id
    group.attrs['visit'] = visit
    group.attrs['side'] = side

    img_ds = group.create_dataset(
        'image', data=img_pixels_crop,
        **hdf5plugin.Blosc2(cname='blosclz', clevel=9,
                            filters=hdf5plugin.Blosc2.SHUFFLE))
    img_ds.attrs['source'] = input_dicom
    if pixel_spacing:
        img_ds.attrs['pixel_spacing'] = pixel_spacing
    if source_pixel_spacing:
        img_ds.attrs['source_pixel_spacing'] = source_pixel_spacing
    if forced_input_pixel_spacing:
        img_ds.attrs['forced_input_pixel_spacing'] = forced_input_pixel_spacing
    if input_pixel_spacing:
        img_ds.attrs['input_pixel_spacing'] = input_pixel_spacing
    img_ds.attrs['intensity_offset'] = intensity_offset
    img_ds.attrs['intensity_slope'] = intensity_slope
    img_ds.attrs['crop_offset_y'] = offset_y
    img_ds.attrs['crop_offset_x'] = offset_x
    img_ds.attrs['femoral_head_radius'] = femoral_head_radius
    img_ds.attrs['sourcil_radius'] = sourcil_radius

    seg_ds = group.create_dataset(
        'segmentation', data=segmentation_crop.astype('uint8'),
        **hdf5plugin.Blosc2(cname='blosclz', clevel=9,
                            filters=hdf5plugin.Blosc2.SHUFFLE))
    seg_ds.attrs['labels'] = json.dumps(LABELS)


parser = argparse.ArgumentParser()
parser.add_argument('--input-csv', metavar='CSV', nargs='+', required=True,
                    help='list of subjects and images as CSV')
parser.add_argument('--images-path', metavar='PATH', required=True,
                    help='base directory for the images')
parser.add_argument('--points-path', metavar='PATH', required=True,
                    help='base directory for the point files')
parser.add_argument('--manual-path', metavar='PATH', required=True,
                    help='base directory for the manual JSON files')
parser.add_argument('--output', metavar='HDF5', required=True,
                    help='output HDF5 file')
parser.add_argument('--visit', metavar='VISIT', default='T00',
                    help='default visit if not given in CSV')
parser.add_argument('--target-pixel-spacing', metavar='SPACING', type=float,
                    help='resample image to target spacing (mm/pixel)')
parser.add_argument('--crop-size', metavar='PIXELS', type=int,
                    help='crop the hips to the required size')
parser.add_argument('--skip-missing-files', action='store_true',
                    help='ignore file-not-found errors')
parser.add_argument('--debug-subset', action='store_true',
                    help='only process a few subjects for debugging')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--num-workers', metavar='N', type=int, default=1,
                    help='load images in parallel')
args = parser.parse_args()


# list all images to be converted
all_files = []
for csv_filename in args.input_csv:
    df = pd.read_csv(csv_filename)
    df = df.rename(columns=lambda s: s.lower())
    if 'exclude' in df:
        # skip excluded rows
        df = df[df['exclude'] != 1]
    for row in df.to_dict(orient='records'):
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
                'dataset': row.get('dataset') or row.get('cohort_name'),
                'subject_id': row.get('subject_id') or row.get('coach_id'),
                'visit': row.get('visit', args.visit),
                'side': row.get('side') or row.get('measurement_side'),
                'input_dicom': input_dicom,
                'input_points': input_points,
                'input_manual': input_manual,
                'forced_input_pixel_spacing': forced_input_pixel_spacing,
            })
        else:
            print(f'Skipped row with missing file: {input_dicom} or {input_points}')

if args.debug_subset:
    all_files = all_files[:30]


def do_load_single_image(task):
    try:
        return load_single_image(
                **task,
                target_pixel_spacing=args.target_pixel_spacing,
                crop_size_in_pixels=args.crop_size)
    except Exception as e:
        # skip this image for now
        print(f"FAILED TO CONVERT {task['input_dicom']}: ")
        print(task)
        print(e)
        import traceback
        traceback.print_exc()
        print()
        return None


# process all images
with h5py.File(args.output, 'w') as h5:
    # pool for loading images in parallel
    with Pool(args.num_workers) as pool:
        for image_data in tqdm.tqdm(pool.imap_unordered(do_load_single_image, all_files),
                                    total=len(all_files),
                                    disable=args.quiet, desc='Converting'):
            if image_data is not None:
                try:
                    write_single_image(h5, **image_data)
                except Exception as e:
                    print(f"FAILED TO WRITE {image_data['subject_id']}/{image_data['visit']}/{image_data['side']}")
                    print(e)
                    print()
