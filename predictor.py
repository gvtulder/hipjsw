import argparse
import numpy as np

import onnxruntime as ort

import util
import loader
import detect


class Predictor:
    def __init__(self, onnx_model):
        # ort session
        self.sess = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])

    def predict(self, image):
        img = image.pixels[None, None, :, :].astype(np.float32)
        pred = self.sess.run(['prediction'], {'images': img})[0][0]
        labels = np.argmax(pred, axis=0)
        return pred, labels


parser = argparse.ArgumentParser(add_help=False)
# model
parser.add_argument('--segmentation-model', metavar='ONNX',
                    default='checkpoints/checkpoint-19160_10-best-val-loss-epoch=227-step=684.onnx',
                    help='segmentation model (ONNX)')
# output
parser.add_argument('--save-segmentation', metavar='PNG',
                    help='save output segmentation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[parser, loader.parser, detect.parser])
    args = parser.parse_args()

    predictor = Predictor(args.segmentation_model)

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
