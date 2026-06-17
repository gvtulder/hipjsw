import argparse
import json
import numpy as np
import os.path
import sys
import tqdm
import traceback
import logging
from pqdm.processes import pqdm

import jsw_measurement
import measurement_utils as u

import loader
import predictor

logging.getLogger('matplotlib').setLevel(logging.CRITICAL)


def process(predictor, measurer, input_dicom, input_points, side, scan_id, args):
    img_data, _, segmentation = predictor.load_and_predict(input_dicom, input_points, side)
    image = img_data['img_pixels_crop']
    crop_offset_mm = [img_data['crop_offset_y_mm'], img_data['crop_offset_x_mm']]

    # compute statistics
    measurement, trace = measurer.measure(segmentation)
    trace['crop_shape'] = image.shape
    trace['crop_offset_mm'] = crop_offset_mm

    if args.show_plots or args.output_plots:
        if 'overview' in args.plot_types:
            # plot overview
            jsw_plot.plot_large_overview(image, segmentation, trace, title=scan_id)

            if args.show_plots:
                plt.show()
            if args.output_plots:
                plt.savefig(f'{args.output_plots}{scan_id.replace("/", "-")}.png')
            plt.close()

        if 'segmeas' in args.plot_types:
            # close-up with segmentation and contour
            js_mask = (segmentation == jsw_plot.LABEL_JOINT_SPACE)
            js_mask_object = u.select_largest_object(js_mask)
            plt.figure(figsize=(4, 3))
            jsw_plot.plot_image_crop(image, js_mask_object, cmap='gray', pixel_spacing=trace['pixel_spacing'])
            jsw_plot.plot_image_crop(segmentation, js_mask_object, alpha=0.5, pixel_spacing=trace['pixel_spacing'])
            jsw_plot.plot_measurements_on_curves(trace, set_aspect=False)
            jsw_plot.set_lim_to_show_curve(trace['smooth_curve_upper'])
            plt.title(scan_id, fontsize=10)
            if args.output_plots:
                plt.savefig(f'{args.output_plots}{scan_id.replace("/", "-")}-SEG.png')
            plt.close()

    if args.output_trace:
        js_mask = (segmentation == jsw_plot.LABEL_JOINT_SPACE)
        js_mask_object = u.select_largest_object(js_mask)

        np.savez_compressed(
            f'{args.output_trace}{scan_id.replace("/", "-")}.npz',
            title=scan_id,
            measurement=measurement,
            trace=trace,
            image=image,
            segmentation=segmentation,
            js_mask_object=js_mask_object,
            crop_offset_mm=crop_offset_mm,
        )

    return measurement, trace


def compute_measurements(predictor, measurer, input_dicom, input_points, side, scan_id, args):
    # load, segment, process image
    err = None
    try:
        measurement, trace = \
            process(predictor, measurer, input_dicom, input_points, side, scan_id, args)
    except Exception as e:
        print()
        print(f'FAILED processing {scan_id}')
        print(e)
        traceback.print_exc()
        measurement = {}
        err = str(e)

    # CSV output
    csv_row = {
        'input_dicom': input_dicom,
        'input_points': input_points,
        'side': side,
        'scan_id': scan_id,
        **{f'jsw {k}': v.item() for k, v in measurement.items() if v.ndim == 0},
        'sourcil length': (np.max(measurement['profile_length']) if err is None else 0),
        **({'error': str(err)} if err is not None else {}),
    }

    # add measurement coordinates
    if err is None:
        for meas_key in [
            'minimum',
            'medial',
            'central',
            'lateral',
        ]:
            for contour_key in [
                'sourcil',
                'femur',
            ]:
                if side == 'left':
                    # x coordinate, horizontal flip of cropped area
                    csv_row[f'seg_{meas_key}_JSW_idx_{contour_key}_x'] = \
                        trace['crop_shape'][0] * trace['pixel_spacing'] - \
                        trace['measurement_points'][f'{contour_key} {meas_key}'][1] + \
                        trace['crop_offset_mm'][1]
                else:
                    # x coordinate
                    csv_row[f'seg_{meas_key}_JSW_idx_{contour_key}_x'] = \
                        trace['measurement_points'][f'{contour_key} {meas_key}'][1] + trace['crop_offset_mm'][1]
                # y coordinate
                csv_row[f'seg_{meas_key}_JSW_idx_{contour_key}_y'] = \
                    trace['measurement_points'][f'{contour_key} {meas_key}'][0] + trace['crop_offset_mm'][0]

    return {
        'csv': csv_row,
    }


parser = argparse.ArgumentParser(add_help=False)
# input (see also predictor args)
parser.add_argument('--input-csv', metavar='CSV',
                    help='input image list in CSV format')
parser.add_argument('--input-dicom', metavar='DCM',
                    help='input image in DICOM or JPEG format')
parser.add_argument('--input-points', metavar='PTS',
                    help='points file')
parser.add_argument('--side', metavar='SIDE', choices=['left', 'right'],
                    help='side')
parser.add_argument('--scan-id', metavar='SCANID',
                    help='optional scan ID for filenames and plots')
parser.add_argument('--pixel-spacing', metavar='SPACING', type=float,
                    default=0.2,
                    help='resample image to target spacing (mm/pixel)')
parser.add_argument('--crop-size', metavar='PIXELS', type=int,
                    default=512,
                    help='crop the hips to the required size')
# outputs
parser.add_argument('--show-plots', action='store_true',
                    help='show plots')
parser.add_argument('--output-plots', metavar='DIR',
                    help='save measurement images as PNG')
parser.add_argument('--plot-types', metavar='PLOT', nargs='+',
                    choices=['overview', 'segmeas'], default=['overview'],
                    help='the type of plots to generate')
parser.add_argument('--output-csv', metavar='CSV',
                    help='save measurements as CSV')
parser.add_argument('--output-trace', metavar='NPZ',
                    help='save measurement trace objects')
parser.add_argument('--print-json', action='store_true',
                    help='print measurements as JSON')
# parallel processing
parser.add_argument('--num-workers', metavar='N', type=int,
                    help='enable parallel processing with N workers')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[predictor.parser, parser])
    args = parser.parse_args()

    # initialize measurement model
    measurer = jsw_measurement.JointSpaceFromSegmentation(pixel_spacing=args.pixel_spacing)

    # conditional imports
    if args.show_plots or args.output_plots:
        import matplotlib.pyplot as plt
        import jsw_plot
    if args.output_csv:
        import pandas as pd

    # segmentation model
    predictor = predictor.Predictor(args.checkpoint, args.model_args,
                                    args.pixel_spacing, args.crop_size, args.device)

    # process images
    all_measurements_csv = []

    if args.input_csv is None:
        input_list = [{
            'input_dicom': args.input_dicom,
            'input_points': args.input_points,
            'side': args.side,
            'scan_id': args.scan_id,
        }]
    else:
        assert args.input_dicom is None, 'input_dicom is incompatible with input_csv'
        assert args.input_points is None, 'input_points is incompatible with input_csv'
        assert args.side is None, 'side is incompatible with input_csv'
        assert args.scan_id is None, 'scan_id is incompatible with input_csv'
        input_list = pd.read_csv(args.input_csv).to_dict('records')

    for row in input_list:
        input_dicom = row['input_dicom']
        input_points = row['input_points']
        side = row['side']
        scan_id = row.get('scan_id') or f'{os.path.basename(input_dicom)}-{side}'
        result = compute_measurements(predictor, measurer,
                                      input_dicom, input_points, side, scan_id, args)
        all_measurements_csv.append(result['csv'])

        if args.print_json:
            print(json.dumps(result['csv']))

    if args.output_csv:
        df = pd.DataFrame(all_measurements_csv)
        df.to_csv(args.output_csv, index=False)

