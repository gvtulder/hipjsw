import argparse
import json
import torch

from model import models


def load_model(checkpoint, model_args):
    # load configuration
    with open(model_args, 'r') as f:
        model_args = json.load(f)
    model = models[model_args['model']](lr=model_args['lr'], num_classes=model_args['num_classes'])
    # load checkpoint
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model.net


parser = argparse.ArgumentParser(add_help=False)
# model
parser.add_argument('--checkpoint', metavar='CHECKPOINT', required=True,
                    help='checkpoint')
parser.add_argument('--model-args', metavar='JSON', required=True,
                    help='argument file for the model')
parser.add_argument('--crop-size', metavar='PIXELS', type=int,
                    default=512,
                    help='crop the hips to the required size')
parser.add_argument('--opset', type=int, default=17)
parser.add_argument('--output-onnx', metavar='ONNX', required=True)
args = parser.parse_args()

model = load_model(args.checkpoint, args.model_args)

example_input = torch.zeros((1, 1, args.crop_size, args.crop_size), dtype=torch.float32)
with torch.no_grad():
    onnx_program = torch.onnx.export(
        model,
        example_input,
        opset_version=args.opset,
        input_names=['images'],
        output_names=['prediction'],
        do_constant_folding=True,
        dynamic_axes={'images': {0: 'batch'}},
        dynamo=True,
    )
onnx_program.save(args.output_onnx)

