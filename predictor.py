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

import util
import loader
import detect

from model import models


class Predictor:
    def __init__(self, checkpoint, model_args, device):
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

    def predict(self, image):
        # convert image to torch
        img = torch.tensor(image.pixels, dtype=torch.float32, device=self.model.device)

        # predict segmentation
        with torch.no_grad():
            pred = self.model.predict(img[None, None, :, :])[0]
            pred = pred.detach().numpy()
            labels = np.argmax(pred, axis=0)

        return pred, labels


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
    parser = argparse.ArgumentParser(parents=[parser, loader.parser, detect.parser])
    args = parser.parse_args()

    predictor = Predictor(args.checkpoint, args.model_args, args.device)

    image_input = loader.load_dicom_image(args.input_dicom)
    hip_detections = detect.detect_from_args(args, image_input)

    cropper = loader.Cropper(args.pixel_spacing, args.crop_size)
    for side, hip_detection in hip_detections.items():
        image_cropped, stats = cropper.process(image_input, hip_detection, side)
        pred, labels = predictor.predict(image_cropped)

        if args.save_image:
            util.save_grayscale_image(args.save_image.format(side=side), image_cropped.pixels)

        if args.save_segmentation:
            util.save_grayscale_image(args.save_segmentation.format(side=side), labels)
