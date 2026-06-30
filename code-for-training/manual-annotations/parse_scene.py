"""
Convert manual sourcil and femur contours from 3D Slicer scenes to JSON.

Load an image in 3D Slicer and annotate four curves:

* left femur
* left sourcil
* right femur
* right sourcil

then use this script to extract the contours from the .mrb scene file.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import argparse
import numpy as np
import json
import xml.etree
import zipfile

import skimage.draw
import imageio.v2 as imageio
import pydicom
import splines

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input scene (.mrb)', required=True)
parser.add_argument('--image', help='background image')
parser.add_argument('--output-overlay', help='save an image with the annotation overlay')
parser.add_argument('--output-json', help='save the curves as JSON')
args = parser.parse_args()



def parse_curve(doc):
    curve = []
    assert len(doc['markups']) == 1
    markup = doc['markups'][0]
    assert markup['type'] == 'Curve'
    assert markup['coordinateSystem'] == 'LPS'
    assert markup['coordinateUnits'] == 'mm'
    for point in markup['controlPoints']:
        assert len(point['position']) == 3
        assert point['position'][2] == 0.0
        curve.append(point['position'][:2])
    return np.array(curve)


curves = []
scene_spacing = None

with zipfile.ZipFile(args.input, 'r') as zf:
    for member in zf.infolist():
        if member.filename.endswith('.json'):
            with zf.open(member, 'r') as f:
                doc = json.load(f) 
                if len(doc['markups']) == 1 and doc['markups'][0]['type'] == 'Curve':
                    curves.append(parse_curve(doc))
        elif member.filename.endswith('.mrml'):
            with zf.open(member, 'r') as f:
                doc = xml.etree.ElementTree.parse(f)
                volumes = doc.findall('Volume')
                if len(volumes) == 0:
                    volumes = doc.findall('VectorVolume')
                assert len(volumes) == 1, volumes
                assert scene_spacing is None
                scene_spacing = [float(s) for s in volumes[0].attrib['spacing'].split(' ')]
                assert len(scene_spacing) == 3, scene_spacing
                assert scene_spacing[2] == 1
                scene_spacing = np.array(scene_spacing[:2])


assert len(curves) <= 4, f'found {len(curves)} curves'

curve_labels = []

if args.image is not None:
    if args.image.endswith('.jpg'):
        img = imageio.imread(args.image)
        img = np.mean(img.astype(float), axis=2)
    else:
        dcm = pydicom.dcmread(args.image)
        img = dcm.pixel_array
        # pixel_spacing = np.array(dcm.PixelSpacing)
        # print(pixel_spacing)

    mean_curve_length = np.mean([c.shape[0] for c in curves])
    curves = [
        curve / scene_spacing[None, :]
        for curve in curves
    ]

    for curve in curves:
        mu_x, mu_y = np.mean(curve, axis=0)
        print(f'mu_x: {mu_x} mu_y: {mu_y} img.shape: {img.shape[1] / 2}')
        # right hip is left on image
        side = 'right' if mu_x < img.shape[1] / 2 else 'left'
        shape = 'sourcil' if curve.shape[0] < mean_curve_length else 'femur'
        curve_labels.append(f'{side} {shape}')

assert len(curve_labels) == len(set(curve_labels)), curve_labels


COLORS = {
    'left sourcil': [255, 0, 0],
    'right sourcil': [100, 0, 0],
    'left femur': [0, 160, 0],
    'right femur': [0, 64, 0],
}


if args.output_overlay is not None:
    if args.image is not None:
        img = img.astype(float)
        img -= img.min()
        img /= img.max()
        img = (img * 255).astype('uint8')

        img = img[:, :, None].repeat(3, axis=2)

        for curve, curve_label in zip(curves, curve_labels):
            # default 3D Slicer spline
            spline = splines.KochanekBartels(curve)
            curve_spline = spline.evaluate(np.linspace(0, max(spline.grid), len(spline.grid) * 10))

            rr, cc = [], []
            cint = np.round(curve_spline).astype(int)
            for i in range(len(curve_spline) - 1):
                r, c = skimage.draw.line(
                    cint[i, 1], cint[i, 0],
                    cint[i + 1, 1], cint[i + 1, 0],
                )
                rr.extend(r)
                cc.extend(c)
            rr = np.array(rr)
            cc = np.array(cc)

            rr = rr[(rr >= 0) & (rr < img.shape[0])]
            cc = cc[(cc >= 0) & (cc < img.shape[1])]

            img[rr, cc, 0] = COLORS[curve_label][0]
            img[rr, cc, 1] = COLORS[curve_label][1]
            img[rr, cc, 2] = COLORS[curve_label][2]

        imageio.imwrite(args.output_overlay, img)


if args.output_json is not None:
    with open(args.output_json, 'w') as f:
        doc = {}
        for curve, curve_label in zip(curves, curve_labels):
            doc[curve_label] = curve.tolist()
            # default 3D Slicer spline
            spline = splines.KochanekBartels(curve)
            curve_spline = spline.evaluate(np.linspace(0, max(spline.grid), len(spline.grid) * 10))
            doc[f'smooth {curve_label}'] = curve_spline.tolist()
        f.write(json.dumps(doc))

