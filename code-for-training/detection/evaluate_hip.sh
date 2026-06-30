#!/bin/bash

source venv/bin/activate

python tools/evaluate.py \
  --weights runs/train/latest/weights/best_model_state.pt \
  --test_folder data-for-yolo/images/test/ \
  --test_folder_labels data-for-yolo/labels/test/

