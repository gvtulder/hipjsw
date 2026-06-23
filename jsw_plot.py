import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import skimage
import scipy.signal
import imageio

import measurement_utils as u


LABEL_FEMUR = 2
LABEL_JOINT_SPACE = 3
COLOR_OUTLINE = 'tab:blue'
COLOR_JOINT_SPACE = 'tab:orange'
COLOR_FEMUR = 'tab:green'
COLOR_MARKERS = 'tab:red'
COLOR_MARKERS_ALT = 'tab:pink'  # black
COLOR_TEXT = 'black'
STYLE_BBOX = dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)
LINE_WIDTH = 1.5
MARKER_SIZE = 5


def plot_contour(trace, mark_interval=15, set_aspect=True):
    # plot the contour with markers at the given interval
    contour = trace['contour_js']
    if set_aspect:
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', 'datalim')
    plt.plot(contour[:, 1], contour[:, 0], linewidth=LINE_WIDTH, color=COLOR_OUTLINE)
    xlim = plt.xlim()
    offset = (xlim[1] - xlim[0]) * 0.02
    if mark_interval is not None:
        plt.plot(contour[::mark_interval, 1], contour[::mark_interval, 0], 'o', color=COLOR_MARKERS, markersize=MARKER_SIZE)
        for i, p in enumerate(contour[::mark_interval, :]):
            plt.text(
                p[1], p[0] - offset,
                i * mark_interval,
                ha='center',
                color=COLOR_TEXT
            )

def plot_polar_outline(trace, ax=None):
    if ax is None:
        ax = plt.gcf().add_subplot(projection='polar')
    ax.set_theta_offset(-np.pi/2.0)
    ax.plot(trace['corners']['contour_angle'],
            trace['corners']['contour_radius'],
            color=COLOR_OUTLINE, linewidth=LINE_WIDTH)

def plot_corner_detection(trace):
    plt.plot(trace['corners']['contour_radius'], label='radius', linewidth=LINE_WIDTH)
    plt.plot(trace['corners']['contour_radius_diff'], label='radius diff', linewidth=LINE_WIDTH)
    plt.plot(trace['corners']['contour_radius_diff2'], label='radius diff2', linewidth=LINE_WIDTH)
    plt.plot(trace['corners']['midpoint'] % len(trace['corners']['contour_radius']),
             trace['corners']['contour_radius'][trace['corners']['midpoint'] % len(trace['corners']['contour_radius'])],
             's', color=COLOR_MARKERS, markersize=MARKER_SIZE)
    plt.plot(trace['corners_idx'], trace['corners']['contour_radius'][trace['corners_idx']],
             'o', color=COLOR_MARKERS, markersize=MARKER_SIZE)
    for p in trace['corners_idx']:
        plt.axvline(p, alpha=0.1, color=COLOR_MARKERS)
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d°'))
    plt.legend()

def plot_contour_with_corners(trace, set_aspect=True):
    if set_aspect:
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', 'datalim')
    plt.plot(trace['contour_js'][:, 1], trace['contour_js'][:, 0], linewidth=LINE_WIDTH)
    plt.plot(trace['contour_js'][trace['corners_idx'], 1], trace['contour_js'][trace['corners_idx'], 0],
             'o', color=COLOR_MARKERS, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
    xlim = plt.xlim()
    offset = (xlim[1] - xlim[0]) * 0.02
    for idx, p in enumerate(trace['corners_idx']):
        plt.text(trace['contour_js'][p, 1],
                 trace['contour_js'][p, 0] + (-offset if idx in (0, 3) else offset),
                 idx,
                 ha='center', va=('bottom' if idx in (0, 3) else 'top'),
                 color=COLOR_TEXT)

def plot_upper_and_lower_curves(trace, set_aspect=True):
    if set_aspect:
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', 'datalim')
    plt.plot(trace['contour_js'][:, 1], trace['contour_js'][:, 0], alpha=0.2, linewidth=LINE_WIDTH)
    plt.plot(trace['contour_js_upper'][:, 1], trace['contour_js_upper'][:, 0], label='upper', linewidth=LINE_WIDTH)
    plt.plot(trace['contour_femoral_head'][:, 1], trace['contour_femoral_head'][:, 0], label='lower', linewidth=LINE_WIDTH)

def plot_smooth_upper_and_lower_curves(trace, set_aspect=True):
    if set_aspect:
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', 'datalim')
    plt.plot(trace['contour_js'][:, 1], trace['contour_js'][:, 0], alpha=0.2, linewidth=LINE_WIDTH)
    plt.plot(trace['smooth_curve_upper'][:, 1], trace['smooth_curve_upper'][:, 0], label='upper', linewidth=LINE_WIDTH)
    plt.plot(trace['smooth_curve_lower'][:, 1], trace['smooth_curve_lower'][:, 0], label='lower', linewidth=LINE_WIDTH)

def plot_measurements_on_curves(trace, set_aspect=True, show_values=True):
    if set_aspect:
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', 'datalim')
    plt.plot(trace['smooth_curve_upper'][:, 1], trace['smooth_curve_upper'][:, 0],
             label='upper', linewidth=LINE_WIDTH, color=COLOR_JOINT_SPACE)
    plt.plot(trace['smooth_curve_lower'][:, 1], trace['smooth_curve_lower'][:, 0],
             label='lower', linewidth=LINE_WIDTH, color=COLOR_FEMUR)
    xlim = plt.xlim()
    offset = (xlim[1] - xlim[0]) * 0.02
    points = trace['measurement_points']
    for key in ['minimum', 'lateral', 'central', 'medial']:
        a = points[f'sourcil {key}']
        b = points[f'femur {key}']
        d = trace['measurements'][key]
        m = (a + b) / 2
        if key == 'minimum':
            plt.plot(a[1], a[0], 'x', color=COLOR_MARKERS_ALT, markersize=MARKER_SIZE)
            plt.plot(b[1], b[0], 'x', color=COLOR_MARKERS_ALT, markersize=MARKER_SIZE)
            plt.plot([a[1], b[1]], [a[0], b[0]], '-', linewidth=3*LINE_WIDTH, alpha=0.5, color=COLOR_MARKERS_ALT)
            if show_values:
                plt.text(b[1], b[0] + offset, f'{key}\n{d:0.1f}', ha='right', va='top',
                         color=COLOR_TEXT, bbox=STYLE_BBOX)
        else:
            plt.plot(a[1], a[0], 'o', color=COLOR_MARKERS, markersize=MARKER_SIZE)
            plt.plot(b[1], b[0], 'o', color=COLOR_MARKERS, markersize=MARKER_SIZE)
            plt.plot([a[1], b[1]], [a[0], b[0]], '--', linewidth=LINE_WIDTH, color=COLOR_MARKERS)
            if show_values:
                plt.text(m[1] + offset, m[0], f'{key}\n{d:0.1f}', ha='left', va='center',
                         color=COLOR_TEXT, bbox=STYLE_BBOX)

def plot_jsw_profile(trace, measurements=None, flip_lr=False):
    if measurements is None:
        measurements = trace['measurements']
    profile = measurements['profile']
    x = measurements['profile_length']
    plt.plot(x, profile)
    if flip_lr:
        plt.gca().invert_xaxis()
    plt.ylim(0, np.max(profile) * 1.05)
    offset = np.max(profile) * 0.05
    n = profile.shape[0]
    for idx, label, ha in ((0, 'lateral', 'right' if flip_lr else 'left'),
                           (n // 2, 'central', 'center'),
                           (n - 1, 'medial', 'left' if flip_lr else 'right')):
        d = measurements[label]
        plt.plot(x[idx], profile[idx], 'o', color=COLOR_MARKERS)
        plt.text(x[idx], profile[idx] - offset,
                 f'{label}\n{d:0.1f}',
                 color=COLOR_TEXT, ha=ha, va='top')

def plot_jsw_profile_with_radial_thickness(trace):
    profile = trace['measurements']['profile']
    profile_angles = np.rad2deg(-trace['measurements']['profile_angle']) % 360
    plt.plot(profile_angles - profile_angles[0], profile)
    plt.plot(np.rad2deg(trace['ray_profile']['angle']) - profile_angles[0],
             trace['ray_profile']['thickness'])
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d°'))

def plot_image_crop(img, mask, pixel_spacing=1.0, **kwargs):
    im = plt.imshow(img, **kwargs)
    scaling = matplotlib.transforms.Affine2D().scale(pixel_spacing)
    im.set_transform(scaling + plt.gca().transData)
    if mask is not None:
        rmin, rmax, cmin, cmax = u.compute_bbox_with_margin(mask, square=False)
    plt.xlim(cmin * pixel_spacing, cmax * pixel_spacing)
    plt.ylim(rmax * pixel_spacing, rmin * pixel_spacing)
    plt.gca().set_aspect('equal', 'datalim')

def set_lim_to_show_curve(curve, flip_lr=False):
    rcmin = np.min(curve, axis=0)
    rcmax = np.max(curve, axis=0)
    cur_xlim = plt.xlim()
    cur_ylim = plt.ylim()
    if flip_lr:
        plt.xlim(max(cur_xlim[1], rcmax[1]), min(cur_xlim[0], rcmin[1]))
    else:
        plt.xlim(min(cur_xlim[0], rcmin[1]), max(cur_xlim[1], rcmax[1]))
    plt.ylim(max(cur_ylim[0], rcmax[0]), min(cur_ylim[1], rcmin[0]))

def plot_large_overview(image, segmentation, trace, title=None):
    # extract joint space from the segmentation
    js_mask = (segmentation == LABEL_JOINT_SPACE)
    js_mask_object = u.select_largest_object(js_mask)

    # initialize figure with multiple axes
    fig = plt.figure(figsize=(3 * 4, 3 * 4))
    gs = fig.add_gridspec(4, 3)

    # source image with segmentation overlay
    ax = fig.add_subplot(gs[0, 0])
    plot_image_crop(image, js_mask_object, cmap='gray', pixel_spacing=trace['pixel_spacing'])
    plot_image_crop(segmentation, js_mask_object, alpha=0.15, pixel_spacing=trace['pixel_spacing'])
    set_lim_to_show_curve(trace['smooth_curve_upper'])
    plt.title('Segmented input', fontsize=10)

    # corner detection plot
    ax = fig.add_subplot(gs[0, 1])
    plot_corner_detection(trace)
    plt.title('Corner detection', fontsize=10)

    # polar coordinates
    ax = fig.add_subplot(gs[0, 2], projection='polar')
    plot_polar_outline(trace, ax)

    # measurements on segmentation
    ax = fig.add_subplot(gs[1, 0])
    plot_image_crop(segmentation, js_mask_object, alpha=0.5, pixel_spacing=trace['pixel_spacing'])
    plot_measurements_on_curves(trace, set_aspect=False)
    set_lim_to_show_curve(trace['smooth_curve_upper'])
    plt.title('Measurements', fontsize=10)

    # jsw profile
    ax = fig.add_subplot(gs[1, 1])
    plot_jsw_profile(trace)
    plt.title('JSW profile', fontsize=10)

    # jsw profile with ray-based measurements
    ax = fig.add_subplot(gs[1, 2])
    plot_jsw_profile_with_radial_thickness(trace)
    plt.title('JSW profile vs ray-based thickness', fontsize=10)

    # detected corners on outline
    # ax = fig.add_subplot(gs[1, 2])
    # plot_contour_with_corners(trace)
    # plot_smooth_upper_and_lower_curves(trace, set_aspect=False)
    # plt.title('Smoothed curves', fontsize=10)

    # large overview overlaid on image
    ax = fig.add_subplot(gs[2:, :])
    plot_image_crop(image, js_mask_object, cmap='gray', pixel_spacing=trace['pixel_spacing'])
    plot_measurements_on_curves(trace, set_aspect=False)
    set_lim_to_show_curve(trace['smooth_curve_upper'])
    plt.title('JSW measurements and curves', fontsize=10)

    if title:
        plt.suptitle(title)
    plt.tight_layout()

def plot_overview_bonefinder(trace, trace_linear=None, title=None):
    # initialize figure with multiple axes
    fig = plt.figure(figsize=(2 * 4, 2 * 4))
    gs = fig.add_gridspec(2, 2)

    # measurements on segmentation
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect('equal')
    plot_measurements_on_curves(trace, set_aspect=False)
    ax.set_ylim(reversed(ax.get_ylim()))
    plt.title('Measurements', fontsize=10)

    if trace_linear is not None:
        # measurements on segmentation
        ax = fig.add_subplot(gs[0, 1])
        ax.set_aspect('equal')
        plot_measurements_on_curves(trace_linear, set_aspect=False)
        ax.set_ylim(reversed(ax.get_ylim()))
        plt.title('Measurements (linear)', fontsize=10)

    # jsw profile
    ax = fig.add_subplot(gs[1, 1])
    for prof_type in ['smooth', 'linear']:
        plot_jsw_profile(None, trace[f'prof_{prof_type}']['measurements'])
    plt.title('JSW profile', fontsize=10)

    if title:
        plt.suptitle(title)
    plt.tight_layout()

def plot_overview_seg_meas_profile(image, segmentation, trace, title=None, flip_lr=False):
    # extract joint space from the segmentation
    js_mask = (segmentation == LABEL_JOINT_SPACE)
    js_mask_object = u.select_largest_object(js_mask)

    # initialize figure with multiple axes
    fig = plt.figure(figsize=(3 * 4, 3 * 3))
    gs = fig.add_gridspec(3, 3)

    # source image with segmentation overlay
    ax = fig.add_subplot(gs[0, 0])
    plot_image_crop(image, js_mask_object, cmap='gray', pixel_spacing=trace['pixel_spacing'])
    plot_image_crop(segmentation, js_mask_object, alpha=0.15, pixel_spacing=trace['pixel_spacing'])
    set_lim_to_show_curve(trace['smooth_curve_upper'], flip_lr)
    plt.title('Segmented input', fontsize=10)

    # measurements on segmentation
    ax = fig.add_subplot(gs[0, 1])
    plot_image_crop(segmentation, js_mask_object, alpha=0.5, pixel_spacing=trace['pixel_spacing'])
    plot_measurements_on_curves(trace, set_aspect=False)
    set_lim_to_show_curve(trace['smooth_curve_upper'], flip_lr)
    plt.title('Measurements', fontsize=10)

    # jsw profile
    ax = fig.add_subplot(gs[0, 2])
    plot_jsw_profile(trace, flip_lr=flip_lr)
    plt.title('JSW profile', fontsize=10)

    # large overview overlaid on image
    ax = fig.add_subplot(gs[1:, :])
    plot_image_crop(image, js_mask_object, cmap='gray', pixel_spacing=trace['pixel_spacing'])
    plot_measurements_on_curves(trace, set_aspect=False)
    set_lim_to_show_curve(trace['smooth_curve_upper'], flip_lr)
    plt.title('JSW measurements and curves', fontsize=10)

    if title:
        plt.suptitle(title)
    plt.tight_layout()

def save_original_image(filename, input_pixels, segmentation, side, trace, crop_trace):
    # image
    img_rgb = np.repeat(input_pixels[:, :, None], repeats=3, axis=2).astype(float)
    percentiles = np.percentile(img_rgb, [2, 98])
    img_rgb = (img_rgb - percentiles[0]) / (percentiles[1] - percentiles[0])
    img_rgb = np.clip(img_rgb, 0, 1)
    # add overlay, resampled to original size
    seg_rescaled = skimage.segmentation.find_boundaries(segmentation)
    seg_rescaled = skimage.transform.rescale(seg_rescaled.astype(float), 1 / crop_trace['scale'])
    seg_rescaled -= seg_rescaled.min()
    seg_rescaled /= seg_rescaled.max()
    if side == 'left':
        # horizontal flip
        seg_rescaled = seg_rescaled[:, ::-1]
    offset_x = int(crop_trace['crop_offset_x'] / crop_trace['scale'])
    offset_y = int(crop_trace['crop_offset_y'] / crop_trace['scale'])
    seg_for_rgb = np.zeros(img_rgb.shape[:2], dtype=bool)
    seg_for_rgb[
        offset_y:(offset_y + seg_rescaled.shape[0]),
        offset_x:(offset_x + seg_rescaled.shape[1]),
    ] = seg_rescaled
    img_rgb[seg_for_rgb > 0, 0] = 1
    img_rgb[seg_for_rgb > 0, 1] = 0
    img_rgb[seg_for_rgb > 0, 2] = 0
    imageio.imsave(filename, (img_rgb * 255).astype(np.uint8))
