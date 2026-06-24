import argparse
import os.path

from . import bonefinder
from . import hip_detector


class HipDetection:
    def __init__(self, side, center_x, center_y, stats={}):
        self.side = side
        self.center_x = center_x
        self.center_y = center_y
        self.stats = stats

    def scale(self, scale):
        return HipDetection(self.side,
                            int(self.center_x * scale),
                            int(self.center_y * scale),
                            self.stats)

    def __repr__(self):
        return f'HipDetection<{self.side} x={self.center_x} y={self.center_y}>'


def detect_with_bonefinder(points_path, side, image, forced_pixel_spacing=None):
    # load points from Bonefinder
    # for DICOM files with pixel spacing, coordinates are defined in mm
    # for other files, coordinates are defined in pixels and must be mapped to mm
    if image.source_pixel_spacing is not None:
        # the source image has pixel spacing: Bonefinder in mm
        forced_pixel_spacing = None
    if points_path.lower().endswith('_l.pts') or points_path.lower().endswith('_r.pts'):
        points = bonefinder.BonefinderPoints(
            points_path, [side],
            pixel_spacing=forced_pixel_spacing,
        )
    elif points_path.lower().endswith('_rasl.pts'):
        # left hip (right on the image) annotated on mirrored image
        assert side == 'left'
        points = bonefinder.BonefinderPoints(
            points_path, [side],
            # unmirror coordinates
            image.shape[1] * image.pixel_spacing[1],
            pixel_spacing=forced_pixel_spacing,
        )
    else:
        # left and right hip
        points = bonefinder.BonefinderPoints(
            points_path,
            pixel_spacing=forced_pixel_spacing,
        )

    # find center of femoral head
    circles = points.circles_in_pixels(image.pixel_spacing)
    circle = circles[f'{side} femoral head']
    center_y = int(circle['yc'])
    center_x = int(circle['xc'])
    stats = {
        'hip_detection': 'bonefinder',
        'femoral_head_radius': circles[f'{side} femoral head']['r'],
        'sourcil_radius': circles[f'{side} sourcil']['r'],
    }

    return { side: HipDetection(side, center_x, center_y, stats) }


def detect_with_hip_detector(image_input, hip_detector_model):
    detector = hip_detector.HipDetector(hip_detector_model)
    detections = detector.process(image_input.pixels)
    return { side: HipDetection(side, d['center_x'], d['center_y'], d)
             for side, d in detections.items() }


def detect_with_coords(side, center_x, center_y):
    return { side: HipDetection(side, center_x, center_y,
                                { 'hip_detection': 'coords' }) }


def detect_from_args(args, image_input):
    if args.input_points:
        assert args.side is not None
        hip_detections = detect_with_bonefinder(
            args.input_points,
            args.side,
            image_input,
            args.input_pixel_spacing,
        )
    elif args.center_x is not None or args.center_y is not None:
        assert args.center_x is not None
        assert args.center_y is not None
        assert args.side is not None
        hip_detections = detect_with_coords(args.side, args.center_x, args.center_y)
    else:
        hip_detections = detect_with_hip_detector(image_input, args.hip_detector_model)
    return hip_detections


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--input-points', metavar='PTS',
                    help='BoneFinder, points file')
parser.add_argument('--center-x', metavar='PIXELS', type=int,
                    help='center x coordinate of femoral head')
parser.add_argument('--center-y', metavar='PIXELS', type=int,
                    help='center y coordinate of femoral head')
parser.add_argument('--side', metavar='SIDE', choices=['left', 'right'],
                    help='side')
parser.add_argument('--hip-detector-model', metavar='ONNX',
                    default=os.path.join(os.path.dirname(__file__),
                                         'checkpoints/yololite_model_decoded.onnx'),
                    help='path to the hip detector model (ONNX)')
