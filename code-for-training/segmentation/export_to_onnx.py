"""
Utility script to convert a trained PyTorch model to ONNX format.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import argparse
import json
import os
import torch
import types

from model import models

# load experiment settings
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', metavar='CHECKPOINT', required=True,
                    help='checkpoint')
parser.add_argument('--model-args', metavar='JSON', required=True,
                    help='argument file for the model')
parser.add_argument('--input-size', metavar='PIXELS', required=True, type=int,
                    help='input size (width and height)')
parser.add_argument('--output', metavar='ONNX', required=True,
                    help='output file to save ONNX model')
parser.add_argument('--opset', type=int, default=18,
                    help='opset version for the ONNX model')
args = parser.parse_args()
vargs = vars(args)


# load configuration
with open(args.model_args, 'r') as f:
    model_args = json.load(f)
model_args = types.SimpleNamespace(**model_args)

# model
print('Preparing model')
model_params = {}
model = models[model_args.model](lr=model_args.lr, num_classes=model_args.num_classes, **model_params)

# load checkpoint
print('Load checkpoint')
checkpoint = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# export
print('Export model')
B = 1
H = W = args.input_size
dummy = torch.zeros((B, 1, H, W), device='cpu', dtype=torch.float32)

with torch.no_grad():
    torch.onnx.export(
        model.net,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=['images'],
        output_names=['prediction'],
        dynamic_shapes=({0: 'batch_size'},),
        dynamo=True,
        external_data=False,
    )
