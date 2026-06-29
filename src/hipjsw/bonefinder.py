import circle_fit
import numpy as np
from functools import cached_property


# maximum distance between neighboring points on a curve (in mm)
# used as a sanity check for the point numbering
MAX_DISTANCE = 75
# maximum femoral head circle radius (in mm)
MAX_FEMORAL_HEAD_RADIUS = 40
# maximum sourcil circle radius (in mm)
MAX_SOURCIL_RADIUS = 45

# define the curves: right first, then left
NUM_POINTS = 80
SIDES = [ 'right', 'left' ]

# renumbered curves
CURVES = {
    'proximal femur':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                           15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                           28, 29, 30, 31, 32, 33, 34],
    'greater trochanter': [6, 35, 36, 37, 38, 39],
    'posterior wall':     [40, 41, 42, 43, 44],
    'ischium and pubis':  [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                           58, 59, 60],
    'foramen':            [60, 61, 62, 63, 64, 65, 66],
    'acetabular roof':    [67, 68, 69, 70, 71, 72, 73, 74],
    'teardrop':           [75, 76, 77, 78, 79],
}
SUB_CURVES = {
    'femoral head':       [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    'sourcil':            [70, 71, 72, 73, 74],
}

# original curves
CURVES = {
    'proximal femur':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 73, 10, 74, 11, 75,
                           12, 76, 77, 13, 78, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25, 26, 27, 28],
    'greater trochanter': [6, 29, 30, 31, 32, 33],
    'posterior wall':     [34, 35, 36, 37, 38],
    'ischium and pubis':  [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                           51, 52, 53],
    'foramen':            [54, 55, 56, 57, 58, 59, 60],
    'acetabular roof':    [61, 62, 63, 79, 64, 65, 66, 67],
    'teardrop':           [68, 70, 72, 69, 71],
}
SUB_CURVES = {
    'femoral head':       [13, 78, 14, 15, 16, 17, 18, 19, 20, 21],
    'sourcil':            [79, 64, 65, 66, 67],
}


class BonefinderPoints:
    """Container for BoneFinder landmark points for one or both hips.

    This class loads the BoneFinder points file containing hip landmarks in
    in the expected format. See the point definitions above.

    """

    def __init__(self, filename, sides=SIDES, flip_horizontal=None, pixel_spacing=None, sanity_checks=True):
        """Load a BoneFinder points file.

        Parameters
        ----------
        filename : str
            path to the BoneFinder points file
        sides : list of str
            left, right, or both
        flip_horizontal : float or None
            if BoneFinder landmarks refer to a flipped image (e.g., right-as-left),
            give the image width in mm to compute the un-flipped coordinates
        pixel_spacing : float or None
            if the landmark coordinates are stored as pixels, provide pixel spacing
            to convert to mm
        sanity_checks : bool
            if True (default), raise errors when finding unexpectedly large distances
        """
        self.sides = sides
        self._load_points(filename)
        if pixel_spacing is not None:
            self._apply_pixel_spacing(pixel_spacing)
        if flip_horizontal:
            self._flip_horizontal(flip_horizontal)
        if sanity_checks:
            self._check_points()

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx]

    def __str__(self):
        s = ['BonefinderPoints:']
        minx, miny, maxx, maxy = self.bounding_box
        s.append(f'  Bounds: [{minx}, {miny}] to [{maxx}, {maxy}]')
        s.append('  Circles:')
        for name, circle in self.circles.items():
            s.append(f'  - {name}: [{"  ".join([f"{k}: {v:.2f}" for k, v in circle.items()])}]')
        return '\n'.join(s)

    def _load_points(self, filename):
        # load points from BoneFinder
        # coordinates are defined in mm
        points = []
        with open(filename, 'r') as f:
            # skip until start of points: line with {
            line = f.readline()
            while line and line.strip() != '{':
                line = f.readline()
            points = []
            # read points until end: line with }
            line = f.readline()
            while line and line.strip() != '}':
                points.append([float(i) for i in line.strip().split(' ')])
                line = f.readline()
        self.points = np.array(points)
        assert self.points.shape == (len(self.sides) * NUM_POINTS, 2), \
            f'expected ({len(self.sides) * NUM_POINTS}, 2) coordinates, found {self.points.shape}'

    def _check_points(self):
        # sanity check
        max_distance = 0
        for side_idx, side in enumerate(self.sides):
            offset = side_idx * NUM_POINTS
            for name, curve in CURVES.items():
                for a, b in zip(curve[:-1], curve[1:]):
                    distance = np.sqrt(((self.points[a + offset] - self.points[b + offset]) ** 2).sum())
                    assert distance < MAX_DISTANCE, \
                           f'found unexpectedly large distance ({distance:0.2f}mm) in {name}: incorrect point numbering?'
            femoral_head_radius = self.circles[f'{side} femoral head']['r']
            assert femoral_head_radius < MAX_FEMORAL_HEAD_RADIUS, \
                   f'unexpectedly large femoral head ({femoral_head_radius:0.2f}mm): incorrect point numbering?'
            sourcil_radius = self.circles[f'{side} femoral head']['r']
            assert sourcil_radius < MAX_SOURCIL_RADIUS, \
                   f'unexpectedly large acetabular roof ({sourcil_radius:0.2f}mm): incorrect point numbering?'

    def _flip_horizontal(self, image_width):
        self.points[:, 0] = image_width - self.points[:, 0]

    def _apply_pixel_spacing(self, pixel_spacing):
        # assuming that the BoneFinder output is in pixels, map to millimeters
        self.points *= pixel_spacing

    @cached_property
    def bounding_box(self):
        """Compute the bounding box containing all landmarks.

        Returns
        -------
        list of int
            returns [min x, min y, max x, max y]

        """
        return [*self.points.min(axis=0), *self.points.max(axis=0)]

    @cached_property
    def circles(self):
        """Fit circles to the femoral head and sourcil.

        Returns
        -------
        dict of dicts
            keys are "left femoral head", "left sourcil", etc.,
            for each circle, provides { xc, yc, r, sigma } coordinates, radius,
            and circle-fitting error, all in mm

        """
        circles = {}
        for side_idx, side in enumerate(self.sides):
            offset = side_idx * NUM_POINTS
            for name, curve in SUB_CURVES.items():
                xc, yc, r, sigma = circle_fit.taubinSVD(self.points[np.array(curve) + offset])
                circles[f'{side} {name}'] = {'xc': xc, 'yc': yc, 'r': r, 'sigma': sigma}
        return circles

    @cached_property
    def curves(self):
        """Return the coordinates of curves.

        Returns
        -------
        dict of numpy arrays
            keys are "left proximal femur" etc., values are N x 2 coordinate arrays (in mm)
        """
        # return curve coordinates
        curves = {}
        for side_idx, side in enumerate(self.sides):
            offset = side_idx * NUM_POINTS
            for name, curve in [*CURVES.items(), *SUB_CURVES.items()]:
                curves[f'{side} {name}'] = self.points[np.array(curve) + offset]
        return curves

    def circles_in_pixels(self, pixel_spacing):
        """Returns the circles, but in pixels."""
        assert pixel_spacing[0] == pixel_spacing[1], 'expecting isotropic pixel spacing'
        return { name: { 'xc': circle['xc'] / pixel_spacing[0],
                         'yc': circle['yc'] / pixel_spacing[1],
                         'r': circle['r'] / pixel_spacing[0] }
                 for name, circle in self.circles.items() }

    def curves_in_pixels(self, pixel_spacing):
        """Returns the curves, but in pixels."""
        return { name: curve / pixel_spacing
                 for name, curve in self.curves.items() }

    def plot(self, img_pixels=None, pixel_spacing=1):
        """Plot the image with landmark points overlay."""
        import matplotlib.pyplot as plt

        # plot the image with superimposed curves
        plt.figure(figsize=(13, 8))
        if img_pixels is not None:
            plt.imshow(img_pixels, "gray", interpolation="none")
            plt.colorbar()
        else:
            plt.gca().set_aspect('equal')

        # plot curves for right and left
        for side_idx, side in enumerate(self.sides):
            offset = side_idx * NUM_POINTS
            for idx, (name, curve) in enumerate(CURVES.items()):
                color = plt.rcParams["axes.prop_cycle"].by_key()["color"][idx]
                plt.plot(*(self.points[np.array(curve) + offset] / pixel_spacing).transpose(),
                         marker="o", color=color,
                         label=f'{side} {name}',)
        plt.legend(title="Curves")
        plt.tight_layout()
