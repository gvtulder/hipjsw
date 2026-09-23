"""
Compare YOLOLite hip ROI detections with the BoneFinder-based ROIs.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import json
import glob
import os.path
import numpy as np
import tqdm
import pandas
import seaborn
import matplotlib.pyplot as plt

bonefinder_gt_path = 'data-for-yolo/stats/test'
yolo_pred_path = 'YoloLite-Official-Repo/runs/onnx_infer/decoded/3/json'

STANDARD_FEMORAL_HEAD_DIAMETER = 55

bonefinder_gt = glob.glob(f'{bonefinder_gt_path}/*.json')
predictions = glob.glob(f'{yolo_pred_path}/*.json')

results = []

for gt_filename in tqdm.tqdm(bonefinder_gt):
    scan_basename = os.path.basename(gt_filename)
    if scan_basename.endswith('.left.json'):
        side = 'left'
        class_id = 0
    else:
        assert scan_basename.endswith('.right.json')
        side = 'right'
        class_id = 1
    image_name = scan_basename.replace(f'.{side}.json', '')
    pred_filename = os.path.join(yolo_pred_path, f'{image_name}.json')
    if os.path.exists(pred_filename):
        with open(gt_filename, 'r') as f:
            gt = json.load(f)
        with open(pred_filename, 'r') as f:
            pred = json.load(f)
        # print(image_name, pred_filename)
        # print(gt)
        # print(pred)

        # take first prediction (best score)
        detection = None
        for det in pred['detections']:
            if det['class_id'] == class_id:
                detection = det
                break

        if detection is None:
            results.append({'scan': image_name, 'side': side})
            # print('NOT FOUND')
            continue

        bbox = np.array(detection['bbox_xyxy'])

        # compute estimated pixel spacing
        gt_pixel_spacing = np.mean(gt['input_pixel_spacing'])
        pred_pixel_spacing = \
            STANDARD_FEMORAL_HEAD_DIAMETER / np.mean(np.abs(bbox[[0, 1]] - bbox[[2, 3]]))
        # computed on a standard 0.2mm spacing -> map back to original input spacing
        resampled_pixel_spacing = np.mean(gt['pixel_spacing'])
        pred_pixel_spacing *= np.mean(gt['input_pixel_spacing']) / resampled_pixel_spacing
        # print(f'PIXEL SPACING: gt {gt_pixel_spacing:0.3f} pred {pred_pixel_spacing:0.3f}')

        # compute center of bbox / femoral head in mm
        gt_center_x = gt['femoral_head_circle']['xc'] * resampled_pixel_spacing
        gt_center_y = gt['femoral_head_circle']['yc'] * resampled_pixel_spacing
        pred_center_x = np.mean(bbox[[0, 2]]) * resampled_pixel_spacing
        pred_center_y = np.mean(bbox[[1, 3]]) * resampled_pixel_spacing
        center_dist = np.linalg.norm(np.array([gt_center_x, gt_center_y]) - np.array([pred_center_x, pred_center_y]))
        # print(f'CENTER: gt ({gt_center_x:0.1f},{gt_center_y:0.1f}) pred ({pred_center_x:0.1f},{pred_center_y:0.1f}) dist {center_dist:0.1f} mm')

        results.append({
            'scan': image_name,
            'side': side,
            'gt_pixel_spacing': gt_pixel_spacing,
            'pred_pixel_spacing': pred_pixel_spacing,
            'gt_center_x': gt_center_x,
            'gt_center_y': gt_center_y,
            'pred_center_x': pred_center_x,
            'pred_center_y': pred_center_y,
            'dist_center': center_dist,
        })

df = pandas.DataFrame(results)

def scan_to_dataset(scan):
    for d in ['FORCE', 'OAI', 'MOST', 'Chingford', 'JoCo', 'RS1', 'RS2', 'RS3', 'SF']:
        if d in scan:
            return d
    if '_T00_AP' in scan:
        return 'CHECK'
    raise Exception(f'unknown dataset {scan}')

df['dataset'] = [scan_to_dataset(scan) for scan in df['scan']]

seaborn.set_theme()

# femoral head center estimates
seaborn.ecdfplot(df, x='dist_center')
plt.title('Error distribution of femoral head center estimates')
plt.xlabel('Error (mm)')
plt.ylabel('Cumulative proportion')
plt.xlim(-0.1, 5)
plt.savefig('femoral-head-errors-202609.png')
plt.savefig('femoral-head-errors-202609.pdf')
plt.close()

# pixel spacing estimates
seaborn.scatterplot(df, x='gt_pixel_spacing', y='pred_pixel_spacing', alpha=0.2)
plt.plot([0.03, 0.35], [0.03, 0.35], '-', linewidth=0.5)
plt.title('Estimated vs true pixel spacing')
plt.xlabel('True pixel spacing (mm/pixel)')
plt.ylabel('Estimated pixel spacing (mm/pixel)')
plt.savefig('pixel-spacing-estimate-errors-202609.png')
plt.savefig('pixel-spacing-estimate-errors-202609.pdf')
plt.close()



df_copy = df.copy()
df_copy['missing'] = ['missing' if np.isnan(d) else '' for d in df['dist_center']]
print(pandas.pivot_table(
    df_copy,
    values=['scan'],
    columns=['missing'],
    aggfunc='count',
    index=['dataset'],
    fill_value='',
))

print(pandas.pivot_table(
    df,
    values=['dist_center'],
    aggfunc=['mean', 'std'],
    index=['dataset'],
))

