import argparse
import os.path

from . import bonefinder
from . import hip_detector


class HipDetection:
    """Hip detection coordinates.

    Attributes
    ----------
    side : "left" or "right"
        the side of the hip
    center_x : float
        the x coordinate (usually in pixels)
    center_y : float
        the y coordinate (usually in pixels)
    stats : dict, optional
        optional extra details from the detection algorithm
    """

    def __init__(self, side, center_x, center_y, stats={}):
        self.side = side
        self.center_x = center_x
        self.center_y = center_y
        self.stats = stats

    def scale(self, scale):
        """Returns a new detection scaled with the given scale factor."""
        return HipDetection(self.side,
                            int(self.center_x * scale),
                            int(self.center_y * scale),
                            self.stats)

    def __repr__(self):
        return f'HipDetection<{self.side} x={self.center_x} y={self.center_y}>'


def detect_with_bonefinder(points_path, side, image, forced_pixel_spacing=None):
    """Detect hips given a BoneFinder points file.

    This method detects hips in a BoneFinder file, determined by fitting a
    circle to the points on the femoral head and using the center of this
    circle (i.e., the center of the femoral head) as the detected location.

    This requires a very specific order of points in the BoneFinder points
    file: see the bonefinder.py file.

    Parameters
    ----------
    points_path : str
        path to the BoneFinder points file
    side : "left" or "right"
        the side of the hip to extract
    image : ImageWithSpacing
        the image corresponding to the points file
    forced_pixel_spacing : float or None
        the expected input pixel spacing, for images that do not contain pixel spacing

    Returns
    -------
    a dict of { side: HipDetection }
        detections can be empty, "left", "right", or both

    """
    # load points from Bonefinder
    # for DICOM files with pixel spacing, coordinates are defined in mm
    # for other files, coordinates are defined in pixels and must be mapped to mm
    if image.pixel_spacing_source == 'file':
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
        'source': 'bonefinder',
        'femoral_head_radius': circles[f'{side} femoral head']['r'],
        'sourcil_radius': circles[f'{side} sourcil']['r'],
    }

    return { side: HipDetection(side, center_x, center_y, stats) }


def detect_with_hip_detector(image_input, hip_detector_model):
    """Detect hips using a hip detection model.

    This method detects the left and/or right hip in an image using a
    trained HipDetector model (a YOLO-style approach).

    Parameters
    ----------
    image_input : ImageWithSpacing
        the image corresponding to the points file
    hip_detector_model : str
        path to a hip detection model in ONNX format

    Returns
    -------
    a dict of { side: HipDetection }
        detections can be empty, "left", "right", or both

    """
    detector = hip_detector.HipDetector(hip_detector_model)
    detections = detector.process(image_input.pixels)
    return { side: HipDetection(side, d['center_x'], d['center_y'],
                                {'source': 'detector', **d})
             for side, d in detections.items() }


def detect_with_coords(side, center_x, center_y):
    """Convert the given coordinates to a HipDetection object.

    Parameters
    ----------
    side : "left" or "right"
        the side of the hip to extract
    center_x : float
        the x coordinate of the center of the femoral head in pixels
    center_y : float
        the y coordinate of the center of the femoral head in pixels

    Returns
    -------
    a dict of { side: HipDetection }
        side is "left" or "right", depending on input

    """
    return { side: HipDetection(side, center_x, center_y,
                                { 'source': 'coords' }) }


def detect_from_args(args, image_input):
    """Detect hips from an input image given input arguments.

    Based on the arguments, this method loads hip detections from BoneFinder,
    given pre-set coordinates, or using the hip detection model.

    * ``--side`` and ``--input-points``: load from BoneFinder points file
    * ``--side`` and ``--center-x`` and ``--center-y``: use given coordinates
    * Otherwise: use the hip detection model

    Parameters
    ----------
    args : argparse.Namespace
        command-line arguments (see above)
    image_input : ImageWithSpacing
        the input image

    a dict of { side: HipDetection }
        detections can be empty, "left", "right", or both

    """
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
group = parser.add_argument_group('Hip detection')
group.add_argument('--input-points', metavar='PTS',
                   help='BoneFinder, points file')
group.add_argument('--center-x', metavar='PIXELS', type=int,
                   help='center x coordinate of femoral head')
group.add_argument('--center-y', metavar='PIXELS', type=int,
                   help='center y coordinate of femoral head')
group.add_argument('--side', metavar='SIDE', choices=['left', 'right'],
                   help='side (left/right)')
group.add_argument('--hip-detector-model', metavar='ONNX',
                   default=os.path.join(os.path.dirname(__file__),
                                        'checkpoints/yololite_model_decoded.onnx'),
                    help='path to the hip detector model (ONNX)')
