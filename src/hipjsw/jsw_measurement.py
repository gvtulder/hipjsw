import numpy as np
import skimage
import scipy.signal
import scipy.stats

from . import measurement_utils as u


class JointSpaceFromSegmentation:
    """Utility class for measuring hip joint space with from a segmentation.

    Attributes
    ----------
    pixel_spacing : float
        the pixel spacing of the segmentation
    """

    def __init__(self, pixel_spacing=1.0):
        self.pixel_spacing = pixel_spacing

        # label definitions in the segmentation output
        self.joint_space_label = 3
        self.femur_label = 2
        self.sourcil_label = 4

        # algorithm parameters
        self.corner_detection_window = 10
        self.femur_max_distance_factor = 2
        self.curve_smoothness = 0.2

        self.upper_max_dist_to_spline = 0.5
        self.max_spline_fitting_iter = 30

    def measure(self, segmentation):
        """Measure the joint space on the segmented hip image.

        Given a segmented hip joint, this algorithm detects the femur and sourcil curves,
        then measures the space at various locations.

        Parameters
        ----------
        segmentation : numpy array
            the segmented image with class labels

        Returns
        -------
        measurements : dict
            the main JSW measurements
        trace : dict
            additional and intermediate measurements for debugging

        """
        trace = {}
        trace['pixel_spacing'] = self.pixel_spacing

        # obtain contours for joint space and femur
        contour_js = self.segmentation_to_outline(segmentation, self.joint_space_label) * self.pixel_spacing
        contour_femur = self.segmentation_to_outline(segmentation, self.femur_label) * self.pixel_spacing
        contour_sourcil = self.segmentation_to_outline(segmentation, self.sourcil_label) * self.pixel_spacing
        trace['contour_js'] = contour_js
        trace['contour_femur'] = contour_femur
        trace['contour_sourcil'] = contour_sourcil

        # compute the coordinates of the center of the femoral head
        # (in this case, this is equal to the center of the image)
        femoral_head_center = (np.array(segmentation.shape) // 2) * self.pixel_spacing
        trace['femoral_head_center'] = femoral_head_center

        # find the four corners of the joint space shape
        corners_idx, trace['corners'] = self.find_corners(contour_js, origin=femoral_head_center, num_corners=4)
        trace['corners_idx'] = corners_idx

        # split the joint space contour at the corners
        contour_js_upper, contour_js_lower = self.find_morphological_contours(segmentation)
        trace['contour_js_upper'] = contour_js_upper
        trace['contour_js_lower'] = contour_js_lower

        # select the nearby part of the femur curve
        mean_distance = u.mean_curve_distance(contour_js_upper, contour_js_lower)
        contour_femoral_head = self.select_nearby_curve(contour_femur, contour_js_lower, self.femur_max_distance_factor * mean_distance)
        trace['contour_femoral_head'] = contour_femoral_head

        # apply curve smoothing
        smooth_curve_upper = self.make_smooth_contour(contour_js_upper)
        smooth_curve_lower = self.make_smooth_contour(contour_femoral_head)
        trace['smooth_curve_upper'] = smooth_curve_upper
        trace['smooth_curve_lower'] = smooth_curve_lower

        # return polar coordinates wrt femoral head center
        trace['smooth_curve_upper_angle'], trace['smooth_curve_upper_dist'] = \
        smooth_curve_upper_angle, smooth_curve_upper_dist = \
            u.polar_coordinates(smooth_curve_upper, femoral_head_center)
        trace['smooth_curve_lower_angle'], trace['smooth_curve_lower_dist'] = \
            u.polar_coordinates(smooth_curve_lower, femoral_head_center)

        # compute ray-based thickness
        trace['ray_profile'] = self.measure_ray_based_thickness(segmentation)

        # compute distances along upper curve
        prof = self.measure_profile(smooth_curve_upper, smooth_curve_lower, femoral_head_center)
        trace['profile'] = prof
        trace['measurement_points'] = prof['measurement_points']
        trace['measurements'] = measurements = prof['measurements']

        return measurements, trace

    def segmentation_to_outline(self, segmentation, label):
        # find the largest object with the given label
        mask = u.select_largest_object(segmentation == label)
        contours = skimage.measure.find_contours(mask)
        # for one object, we expect to find exactly one contour
        # take largest contour if there is more than one
        return contours[np.argmax(c.shape[0] for c in contours)]

    def find_corners(self, contour, origin, num_corners=4):
        # convert to polar coordinates with the femoral head at the origin
        contour_angle, contour_radius = u.polar_coordinates(contour, origin)

        # find a suitable starting position for the analysis:
        # it is inconvenient if the contour starts at one of the corners,
        # so we find a location somewhere in the middle of the upper curve
        midpoint = int(scipy.stats.circmean(
            np.where(contour_radius > np.median(contour_radius))[0],
            high=len(contour_radius)))

        # corners are located at points where the radius changes quickly
        # compute the first and second-order derivatives of the radius
        contour_radius_diff = np.roll(contour_radius, shift=self.corner_detection_window, axis=0) \
                              - np.roll(contour_radius, shift=-self.corner_detection_window, axis=0)
        contour_radius_diff2 = np.roll(contour_radius_diff, shift=self.corner_detection_window, axis=0) \
                               - np.roll(contour_radius_diff, shift=-self.corner_detection_window, axis=0)
        contour_radius_diff2_abs = np.abs(contour_radius_diff2)

        # find the peaks in the second-order derivative
        all_peaks, properties = scipy.signal.find_peaks(
            np.roll(contour_radius_diff2_abs, shift=-midpoint),
            prominence=(None, None),
            width=(None, None),
            threshold=(None, None))
        order_by_prominence = np.argsort(-properties['prominences'])
        peaks = all_peaks[sorted(order_by_prominence[:4])]

        trace = {
            'contour_angle': contour_angle,
            'contour_radius': contour_radius,
            'contour_radius_diff': contour_radius_diff,
            'contour_radius_diff2': contour_radius_diff2,
            'contour_radius_diff2_abs': contour_radius_diff2_abs,
            'peaks': peaks,
            'midpoint': midpoint,
        }

        # map back from shifted to original contour
        return (peaks + midpoint) % len(contour), trace

    def find_morphological_contours(self, segmentation):
        # use morphological operations to find the upper and lower joint space contours

        # use the largest blobs for femur and sourcil
        femur = u.select_largest_object(segmentation == self.femur_label)
        sourcil = u.select_largest_object(segmentation == self.sourcil_label)
        # the joint space might contain more than one object,
        # e.g. in case of bone-to-bone contact
        joint_space = (segmentation == self.joint_space_label)

        # select sourcil pixels directly next to joint space or femur
        upper_boundary = \
            sourcil & \
            skimage.morphology.dilation(femur + joint_space,
                                        footprint=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        # select femur pixels directly next to joint space or sourcil
        lower_boundary = \
            femur & \
            skimage.morphology.dilation(sourcil + joint_space,
                                        footprint=np.array([[0,1,0],[1,1,1],[0,1,0]]))

        # sort points by x coordinate
        upper_contour = np.array(np.where(upper_boundary)).T
        lower_contour = np.array(np.where(lower_boundary)).T
        upper_contour = upper_contour[np.argsort(upper_contour[:, 0])]
        upper_contour = upper_contour[np.argsort(upper_contour[:, 1])]
        lower_contour = lower_contour[np.argsort(lower_contour[:, 0])]
        lower_contour = lower_contour[np.argsort(lower_contour[:, 1])]

        return upper_contour * self.pixel_spacing, lower_contour * self.pixel_spacing

    def fit_curve_to_contour(self, contour, smoothness, max_dist_to_spline, max_iter):
        # initial mask to include only the center of the upper curve
        contour_mask = np.zeros(contour.shape[0], dtype=bool)
        contour_mask[int(contour.shape[0] * 0.20):int(contour.shape[0] * 0.80)] = True

        max_dist = max_dist_to_spline * 2
        # iteratively fit curves and include points until no points are added or removed
        for _ in range(max_iter):
            # print number of points currently matched
            # print(f"{np.sum(contour_mask)}", end=" ")

            # fit new spline to current points
            spl, uu0 = scipy.interpolate.make_splprep(
                contour[contour_mask].T,
                s=np.sum(contour_mask) * smoothness,
                k=3)
            # extrapolate a bit
            uu1 = np.linspace(uu0[0] - 0.1, uu0[-1] + 0.1, contour.shape[0])
            interpolated_contour = np.array(spl(uu1)).T

            # plot curve
            # plt.plot(interpolated_contour[:, 1], interpolated_contour[:, 0], alpha=1)

            # select points that are close to the fitted spline
            dist = u.pointwise_distance_to_curve(contour, interpolated_contour)
            previous_count = np.sum(contour_mask)
            contour_mask = (dist < max_dist_to_spline)

            # check for convergence
            if previous_count == np.sum(contour_mask):
                if max_dist > max_dist_to_spline:
                    # fine-tune with a smaller distance
                    max_dist = max_dist_to_spline
                else:
                    break

        # select all points on the upper contour from start to end,
        # including any outliers in the middle of the curve
        indices = np.where(contour_mask)[0]
        start = min(indices)
        end = max(indices)
        return u.crop_contour(contour, start, end)

    def select_nearby_curve(self, contour, other_contour, max_distance):
        min_dist = np.min(np.linalg.norm(contour[:, None, :] - other_contour[None, :, :], axis=2), axis=1)
        return contour[min_dist < max_distance]

    def make_smooth_contour(self, contour, num=None):
        if num is None:
            num = contour.shape[0]
        spl, u = scipy.interpolate.make_splprep(
            contour.T,
            s=contour.shape[0] * self.curve_smoothness,
            k=3)
        uu1 = np.linspace(u[0], u[-1], num)
        return np.array(spl(uu1)).T

    def make_linear_interpolation(self, contour, num=None):
        if num is None:
            num = contour.shape[0]
        spl, u = scipy.interpolate.make_splprep(
            contour.T,
            s=0, k=1)
        uu1 = np.linspace(u[0], u[-1], num)
        return np.array(spl(uu1)).T

    def measure_ray_based_thickness(self, segmentation):
        # transform around center of image
        seg_js = skimage.transform.warp_polar(segmentation == self.joint_space_label).astype(int)
        # find all non-zero angles
        mask = np.sum(seg_js, axis=1) > 0
        # find first and last non-zero pixel for each angle
        lower = np.argmax(seg_js, axis=1)
        upper = seg_js.shape[1] - np.argmax(seg_js[:, ::-1], axis=1)
        # compute the thickness in pixels along the ray in each angle
        thickness = upper - lower
        thickness[~mask] = 0
        # compute angles to match with profile_angles
        angles = np.deg2rad((np.arange(360) + 270) % 360)
        # return non-zero thickness only
        return {
            'angle': angles[mask],
            'thickness': thickness[mask] * self.pixel_spacing,
        }

    def measure_profile(self, curve_upper, curve_lower, origin):
        # compute distances along upper curve
        pairwise_dist = np.linalg.norm(curve_upper[:, None, :] - curve_lower[None, :, :], axis=2)
        profile = np.min(pairwise_dist, axis=1)
        profile_length = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(curve_upper, axis=0), axis=1))])
        min_idx = np.argmin(profile, axis=0)
        central_idx = curve_upper.shape[0] // 2
        closest_idx = np.argmin(pairwise_dist, axis=1)

        # for points where the upper curve dips below (= has a large y coord)
        # the lower curve, set distance to 0
        profile[curve_upper[:, 0] > curve_lower[closest_idx, 0]] = 0

        # return polar coordinates wrt femoral head center
        curve_upper_angle, curve_upper_dist = u.polar_coordinates(curve_upper, origin)

        # find key points
        measurement_points = {
            'sourcil lateral': curve_upper[0],
            'sourcil central': curve_upper[central_idx],
            'sourcil medial':  curve_upper[-1],
            'sourcil minimum': curve_upper[min_idx],
            'femur lateral':   curve_lower[closest_idx[0]],
            'femur central':   curve_lower[closest_idx[central_idx]],
            'femur medial':    curve_lower[closest_idx[-1]],
            'femur minimum':   curve_lower[closest_idx[min_idx]],
        }

        # compute joint-space width at various points
        measurements = {
            'profile': profile,
            'profile_length': profile_length,
            'profile_angle': curve_upper_angle,
            'lateral': profile[0],
            'central': profile[profile.shape[0] // 2],
            'medial':  profile[-1],
            'minimum': np.min(profile),
        }

        return {
            'min_idx': min_idx,
            'central_idx': central_idx,
            'closest_idx': closest_idx,
            'measurement_points': measurement_points,
            'measurements': measurements,
        }


class JointSpaceFromBonefinder(JointSpaceFromSegmentation):
    """Utility class to measure joint space width given a BoneFinder points file."""

    def measure(self, pts, side='left', interpolation='smooth'):
        """Measure the joint space from a BoneFinder points file.

        Given a BoneFinder landmark points, this algorithm detects the femur and sourcil curves,
        then measures the space at various locations.

        This is mainly useful for comparing the BoneFinder-derived measurements with the
        segmentation-based measurements.

        Parameters
        ----------
        pts : BoneFinder
            the BoneFinder points object
        side : "left" or "right"
            the side of the hip
        interpolation : "smooth" (default) or "linear"
            whether to apply smooth interpolation on the BoneFinder landmark points

        Returns
        -------
        measurements : dict
            the main JSW measurements
        trace : dict
            additional and intermediate measurements for debugging

        """
        trace = {}

        sourcil = pts.curves[f'{side} sourcil'][:, ::-1]
        femoral_head = pts.curves[f'{side} femoral head'][:, ::-1]
        trace['contour_js_upper'] = sourcil
        trace['contour_js_lower'] = femoral_head

        circle = pts.circles[f'{side} femoral head']
        femoral_head_center = np.array(circle['xc'], circle['yc'])
        trace['femoral_head_center'] = circle

        linear_sourcil = self.make_linear_interpolation(sourcil, num=200)
        linear_femoral_head = self.make_linear_interpolation(femoral_head, num=200)

        if interpolation == 'linear':
            smooth_sourcil = linear_sourcil
            smooth_femoral_head = linear_femoral_head
        else:
            assert interpolation == 'smooth'
            smooth_sourcil = self.make_smooth_contour(sourcil, num=200)
            smooth_femoral_head = self.make_smooth_contour(femoral_head, num=200)
        trace['smooth_curve_upper'] = smooth_sourcil
        trace['smooth_curve_lower'] = smooth_femoral_head

        prof_smooth = self.measure_profile(smooth_sourcil, smooth_femoral_head, femoral_head_center)
        prof_linear = self.measure_profile(linear_sourcil, linear_femoral_head, femoral_head_center)
        trace['prof_smooth'] = prof_smooth
        trace['prof_linear'] = prof_linear

        trace['profile'] = prof_smooth
        trace['measurement_points'] = prof_smooth['measurement_points']
        trace['measurements'] = measurements = prof_smooth['measurements']

        return measurements, trace

