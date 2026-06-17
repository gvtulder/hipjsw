import argparse
import numpy as np
import os.path
import imageio
import re
import skimage
import sys

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

def load_single_image(input_dicom, input_points, side,
        target_pixel_spacing, crop_size_in_pixels, forced_input_pixel_spacing=None):

    ########################
    ## IMAGE FROM DICOM/JPG
    ########################
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
    ## CROPPING / FLIPPING
    ########################
    # crop the hip centered on the femoral head
    circles = points.circles_in_pixels(pixel_spacing)
    circle = circles[f'{side} femoral head']
    offset_y = np.clip(int(circle['yc']) - crop_size_in_pixels // 2,
                       0, img_pixels.shape[0] - crop_size_in_pixels)
    offset_x = np.clip(int(circle['xc']) - crop_size_in_pixels // 2,
                       0, img_pixels.shape[1] - crop_size_in_pixels)
    img_pixels_crop = img_pixels[
        offset_y:offset_y + crop_size_in_pixels,
        offset_x:offset_x + crop_size_in_pixels
    ]

    # check cropped size
    assert img_pixels_crop.shape[0] == crop_size_in_pixels, \
        f'incorrect image size after cropping {img_pixels_crop.shape}'
    assert img_pixels_crop.shape[1] == crop_size_in_pixels, \
        f'incorrect image size after cropping {img_pixels_crop.shape}'

    # flip left to right
    if side == 'left':
        img_pixels_crop = img_pixels_crop[:, ::-1]

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
        'side': side,
        'input_dicom': input_dicom,
        'input_points': input_points,
        'target_pixel_spacing': target_pixel_spacing,
        'crop_size_in_pixels': crop_size_in_pixels,
        'img_pixels_crop': img_pixels_crop,
        'pixel_spacing': pixel_spacing,
        'source_pixel_spacing': source_pixel_spacing,
        'forced_input_pixel_spacing': forced_input_pixel_spacing,
        'input_pixel_spacing': input_pixel_spacing,
        'intensity_offset': intensity_offset,
        'intensity_slope': intensity_slope,
        'crop_offset_y': offset_y,
        'crop_offset_x': offset_x,
        'crop_offset_y_mm': offset_y * pixel_spacing[1],
        'crop_offset_x_mm': offset_x * pixel_spacing[0],
        'femoral_head_radius': femoral_head_radius,
        'sourcil_radius': sourcil_radius,
    }


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--input-dicom', metavar='DCM', required=True,
                    help='input image in DICOM or JPEG format')
parser.add_argument('--input-points', metavar='PTS', required=True,
                    help='points file')
parser.add_argument('--side', metavar='SIDE', choices=['left', 'right'], required=True,
                    help='side')
parser.add_argument('--pixel-spacing', metavar='SPACING', type=float,
                    default=0.2,
                    help='resample image to target spacing (mm/pixel)')
parser.add_argument('--crop-size', metavar='PIXELS', type=int,
                    default=512,
                    help='crop the hips to the required size')
parser.add_argument('--save-image', metavar='PNG',
                    help='save cropped image')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_args()

    img_data = load_single_image(
        args.input_dicom,
        args.input_points,
        args.side,
        args.pixel_spacing,
        args.crop_size
    )
    print(img_data)

    if args.save_image:
        img = img_data['img_pixels_crop']
        img = img.astype(float)
        img -= img.min()
        img /= img.max()
        imageio.imsave(args.save_image, (img * 255).astype('uint8'))
