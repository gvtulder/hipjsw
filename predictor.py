import argparse
import numpy as np
import imageio
import json
import os
import types

import torch
import torch.nn
import lightning.pytorch as pl
import monai

import loader

from model import models


class Predictor:
    def __init__(self, checkpoint, model_args, target_pixel_spacing, crop_size, device):
        # load configuration
        with open(model_args, 'r') as f:
            model_args = json.load(f)
        model_args = types.SimpleNamespace(**model_args)

        # set numpy, torch, python.random seeds
        pl.seed_everything(model_args.seed, workers=True, verbose=False)

        # model
        model_params = {}
        model = models[model_args.model](lr=model_args.lr, num_classes=model_args.num_classes, **model_params)

        # load checkpoint
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        if device:
            model = model.to(device)

        self.model = model
        self.target_pixel_spacing = target_pixel_spacing
        self.crop_size = crop_size

    def load_and_predict(self, input_dicom, input_points, side):
        # load image
        img_data = loader.load_single_image(
            input_dicom,
            input_points,
            side,
            self.target_pixel_spacing,
            self.crop_size
        )

        # convert image to torch
        img = img_data['img_pixels_crop']
        img = torch.tensor(img, dtype=torch.float32, device=self.model.device)

        # predict segmentation
        pred = self.model.predict(img[None, None, :, :])[0]
        pred = pred.detach().numpy()
        labels = np.argmax(pred, axis=0)

        return img_data, pred, labels


parser = argparse.ArgumentParser(add_help=False)
# model
parser.add_argument('--checkpoint', metavar='CHECKPOINT', required=True,
                    help='checkpoint')
parser.add_argument('--model-args', metavar='JSON', required=True,
                    help='argument file for the model')
parser.add_argument('--device', metavar='DEVICE', default='cpu',
                    help='run on CPU or GPU?')
# output
parser.add_argument('--save-segmentation', metavar='PNG',
                    help='save output segmentation')


if __name__ == '__main__':
    # load experiment settings
    parser = argparse.ArgumentParser(parents=[loader.parser, parser])
    args = parser.parse_args()

    predictor = Predictor(args.checkpoint, args.model_args,
                          args.pixel_spacing, args.crop_size, args.device)

    img_data, pred, labels = predictor.load_and_predict(args.input_dicom, args.input_points, args.side)

    if args.save_image:
        img = img_data['img_pixels_crop']
        img = img.astype(float)
        img -= img.min()
        img /= img.max()
        imageio.imsave(args.save_image, (img * 255).astype('uint8'))

    if args.save_segmentation:
        labels = labels.astype(float)
        labels -= labels.min()
        labels /= labels.max()
        imageio.imsave(args.save_segmentation, (labels * 255).astype('uint8'))
