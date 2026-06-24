import argparse
import json
import numpy as np
import os.path
import sys
import traceback
import logging
import matplotlib.pyplot as plt
import pandas as pd

from . import __version__
from . import jsw_measurement
from . import jsw_plot
from . import measurement_utils as u

from . import loader
from . import detect
from . import predictor

logging.getLogger('matplotlib').setLevel(logging.CRITICAL)


def process(cropper, predictor, measurer,
            input_dicom, input_points, input_pixel_spacing, side,
            center_x, center_y, scan_id, args):
    # load image
    image_input = loader.load_image(input_dicom, input_pixel_spacing)

    # detect hips
    if input_points is not None:
        assert side is not None
        assert image_input.pixel_spacing is not None
        hip_detections = detect.detect_with_bonefinder(
            input_points,
            side,
            image_input,
            input_pixel_spacing,
        )
        scan_ids = [scan_id] * len(hip_detections)
    elif center_x is not None or center_y is not None:
        assert center_x is not None
        assert center_y is not None
        assert side is not None
        assert image_input.pixel_spacing is not None
        hip_detections = detect.detect_with_coords(side, center_x, center_y)
        scan_ids = [scan_id] * len(hip_detections)
    else:
        assert side is None
        # for image without a known or given pixel spacing,
        # estimate using the femoral head radius
        if image_input.pixel_spacing is None:
            # estimate pixel spacing, then reload
            hip_detections = detect.detect_with_hip_detector(image_input, args.hip_detector_model)
            mean_head_diameter = np.mean([hip.stats['diameter'] for hip in hip_detections.values()])
            estimated_pixel_spacing = args.standard_head_diameter / mean_head_diameter
            print(f'WARNING: No pixel spacing known for {input_dicom}. '+
                  'Estimating based on femoral head diameter: ' +
                  f'expecting {args.standard_head_diameter:0.1f} mm / '+
                  f'found {mean_head_diameter:0.1f} pixels = '+
                  f'estimated spacing {estimated_pixel_spacing:0.3f} mm/pixel.',
                  file=sys.stderr)
            image_input = loader.load_image(input_dicom, estimated_pixel_spacing)
        hip_detections = detect.detect_with_hip_detector(image_input, args.hip_detector_model)
        scan_ids = [f'{scan_id}/{side}' for side in hip_detections]

    # predict for all hips
    results = {}
    for scan_id, (side, hip_detection) in zip(scan_ids, hip_detections.items()):
        # crop and segment
        image_cropped, crop_trace = cropper.process(image_input, hip_detection, side)
        _, segmentation = predictor.predict(image_cropped)

        # compute statistics
        measurement, trace = measurer.measure(segmentation)
        trace['crop'] = crop_trace
        trace['crop_shape'] = image_cropped.shape
        trace['crop_offset_mm'] = [crop_trace['crop_offset_y_mm'],
                                   crop_trace['crop_offset_x_mm']]

        if args.show_plots or args.output_plots:
            if 'overview' in args.plot_types:
                # plot overview
                jsw_plot.plot_large_overview(image_cropped.pixels, segmentation, trace, title=scan_id)

                if args.show_plots:
                    plt.show()
                if args.output_plots:
                    plt.savefig(args.output_plots.format(
                        scan_id=scan_id.replace('/', '-'),
                        side=side,
                        plot='overview',
                    ))
                plt.close()

            if 'detail' in args.plot_types:
                # plot detail with overvew
                jsw_plot.plot_overview_seg_meas_profile(
                    image_cropped.pixels, segmentation, trace, title=scan_id,
                    flip_lr=args.plot_left_right)

                if args.show_plots:
                    plt.show()
                if args.output_plots:
                    plt.savefig(args.output_plots.format(
                        scan_id=scan_id.replace('/', '-'),
                        side=side,
                        plot='detail',
                    ))
                plt.close()

            if 'segmeas' in args.plot_types:
                # close-up with segmentation and contour
                js_mask = (segmentation == jsw_plot.LABEL_JOINT_SPACE)
                js_mask_object = u.select_largest_object(js_mask)
                plt.figure(figsize=(4, 3))
                jsw_plot.plot_image_crop(image_cropped.pixels, js_mask_object, cmap='gray', pixel_spacing=trace['pixel_spacing'])
                jsw_plot.plot_image_crop(segmentation, js_mask_object, alpha=0.5, pixel_spacing=trace['pixel_spacing'])
                jsw_plot.plot_measurements_on_curves(trace, set_aspect=False)
                jsw_plot.set_lim_to_show_curve(trace['smooth_curve_upper'])
                plt.title(scan_id, fontsize=10)
                if args.output_plots:
                    plt.savefig(args.output_plots.format(
                        scan_id=scan_id.replace('/', '-'),
                        side=side,
                        plot='segmeas',
                    ))
                plt.close()

            if 'overlay' in args.plot_types:
                # original image with segmentation overlay
                if args.output_plots:
                    filename = args.output_plots.format(
                        scan_id=scan_id.replace('/', '-'),
                        side=side,
                        plot='overlay',
                    )
                    jsw_plot.save_original_image(filename, image_input.pixels, segmentation,
                                                 side, trace, crop_trace)

        if args.output_trace:
            js_mask = (segmentation == jsw_plot.LABEL_JOINT_SPACE)
            js_mask_object = u.select_largest_object(js_mask)

            np.savez_compressed(
                args.output_trace.format(
                    scan_id=scan_id.replace('/', '-'),
                    side=side,
                ),
                title=scan_id,
                measurement=measurement,
                trace=trace,
                image=image_cropped.pixels,
                segmentation=segmentation,
                js_mask_object=js_mask_object,
            )

        results[side] = (measurement, trace)

    return results


def compute_measurements(cropper, predictor, measurer,
                         input_dicom, input_points, input_pixel_spacing, side,
                         center_x, center_y, scan_id, args):
    # load, segment, process image
    err = None
    try:
        measurements = \
            process(cropper, predictor, measurer,
                    input_dicom, input_points, input_pixel_spacing, side,
                    center_x, center_y, scan_id, args)
    except Exception as e:
        print()
        print(f'FAILED processing {scan_id}')
        print(e)
        traceback.print_exc()
        measurements = {}
        err = str(e)

    csv_rows = []
    for side, (measurement, trace) in measurements.items():
        # CSV output
        csv_row = {
            'input_dicom': input_dicom,
            **({'input_points': input_points} if input_points else {}),
            'side': side,
            'scan_id': scan_id,
            **{f'jsw {k}': v.item() for k, v in measurement.items() if v.ndim == 0},
            'sourcil length': (np.max(measurement['profile_length']) if err is None else 0),
            **({'error': str(err)} if err is not None else {}),
        }
        csv_rows.append(csv_row)

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
                        csv_row[f'{meas_key}_JSW_{contour_key}_x'] = \
                            trace['crop_shape'][1] * trace['pixel_spacing'] - \
                            trace['measurement_points'][f'{contour_key} {meas_key}'][1] + \
                            trace['crop_offset_mm'][1]
                    else:
                        # x coordinate
                        csv_row[f'{meas_key}_JSW_{contour_key}_x'] = \
                            trace['measurement_points'][f'{contour_key} {meas_key}'][1] + trace['crop_offset_mm'][1]
                    # y coordinate
                    csv_row[f'{meas_key}_JSW_{contour_key}_y'] = \
                        trace['measurement_points'][f'{contour_key} {meas_key}'][0] + trace['crop_offset_mm'][0]

    return {
        'csv': csv_rows,
    }


parser = argparse.ArgumentParser(add_help=False)
# input (see also loader and predictor args)
parser.add_argument('--input-csv', metavar='CSV',
                    help='input image list in CSV format')
parser.add_argument('--images-path', metavar='PATH',
                    help='the path for images listed in the CSV')
parser.add_argument('--points-path', metavar='PATH',
                    help='the path for points files listed in the CSV')
parser.add_argument('--scan-id', metavar='SCANID',
                    help='optional scan ID for filenames and plots')
# hip size estimation
parser.add_argument('--standard-head-diameter', metavar='MM', default=55, type=float,
                    help='expected diameter of the femoral head, used to estimate unknown pixel spacing')
# outputs
parser.add_argument('--show-plots', action='store_true',
                    help='show plots')
parser.add_argument('--output-plots', metavar='DIR',
                    help='save measurement images as PNG')
parser.add_argument('--plot-left-right', action='store_true',
                    help='use the original left/right orientation (default: flip left hips to right)')
parser.add_argument('--plot-types', metavar='PLOT', nargs='+',
                    choices=['overview', 'segmeas', 'overlay', 'detail'],
                    default=['detail'],
                    help='the type of plots to generate')
parser.add_argument('--output-csv', metavar='CSV',
                    help='save measurements as CSV')
parser.add_argument('--output-trace', metavar='NPZ',
                    help='save measurement trace objects')
parser.add_argument('--print-json', action='store_true',
                    help='print measurements as JSON')
parser.add_argument('--version', action='store_true',
                    help='print version and exit')

def hipjsw_cli():
    cli_parser = argparse.ArgumentParser(parents=[predictor.parser, detect.parser,
                                                  loader.parser, parser])
    args = cli_parser.parse_args()

    if args.version:
        print(f'hipjsw version {__version__}')
        sys.exit()

    # initialize predictor and measurement model
    cropper = loader.Cropper(args.pixel_spacing, args.crop_size)
    predictor_model = predictor.Predictor(args.segmentation_model)
    measurer = jsw_measurement.JointSpaceFromSegmentation(pixel_spacing=args.pixel_spacing)

    # process images
    all_measurements_csv = []

    if args.input_csv is None:
        assert args.input_dicom is not None, 'no input file specified'
        input_list = [{
            'input_dicom': args.input_dicom if args.images_path is None else os.path.join(args.images_path, args.input_dicom),
            'input_points': args.input_points if args.points_path is None else os.path.join(args.images_path, args.input_points),
            'input_pixel_spacing': args.input_pixel_spacing,
            'center_x': args.center_x,
            'center_y': args.center_y,
            'side': args.side,
            'scan_id': args.scan_id,
        }]
    else:
        assert args.input_dicom is None, 'input_dicom is incompatible with input_csv'
        assert args.input_points is None, 'input_points is incompatible with input_csv'
        assert args.input_pixel_spacing is None, 'input_pixel_spacing is incompatible with input_csv'
        assert args.center_x is None, 'center_x is incompatible with input_csv'
        assert args.center_y is None, 'center_y is incompatible with input_csv'
        assert args.side is None, 'side is incompatible with input_csv'
        assert args.scan_id is None, 'scan_id is incompatible with input_csv'
        input_list = pd.read_csv(args.input_csv).to_dict('records')

    for row in input_list:
        input_dicom = row['input_dicom']
        input_points = row.get('input_points')
        input_pixel_spacing = row.get('input_pixel_spacing') or args.input_pixel_spacing
        center_x = row.get('center_x')
        center_y = row.get('center_y')
        side = row.get('side')
        scan_id = row.get('scan_id') or os.path.basename(input_dicom)
        result = compute_measurements(cropper, predictor_model, measurer,
                                      input_dicom, input_points, input_pixel_spacing, side,
                                      center_x, center_y, scan_id, args)
        all_measurements_csv += result['csv']

    if args.output_csv:
        df = pd.DataFrame(all_measurements_csv)
        df.to_csv(args.output_csv, index=False)

    if args.print_json:
        print(json.dumps(all_measurements_csv, indent=True))


if __name__ == '__main__':
    hipjsw_cli()
