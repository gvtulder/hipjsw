#!/bin/bash

source venv/bin/activate

python export/export_onnx.py \
  --weights runs/train/latest/weights/best_model_state.pt \
  --img-size 640 \
  --device cpu \
  --simplify
