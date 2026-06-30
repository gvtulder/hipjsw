#!/bin/bash

source venv/bin/activate

python export/infer_onnx.py \
  --model runs/export/3/model_decoded.onnx \
  --img_size 640 \
  --inter 5 --intra 5 \
  --img_dir ../data-for-yolo/images/train/

python export/infer_onnx.py \
  --model runs/export/3/model_decoded.onnx \
  --img_size 640 \
  --inter 5 --intra 5 \
  --img_dir ../data-for-yolo/images/val/

python export/infer_onnx.py \
  --model runs/export/3/model_decoded.onnx \
  --img_size 640 \
  --inter 5 --intra 5 \
  --img_dir ../data-for-yolo/images/test/

