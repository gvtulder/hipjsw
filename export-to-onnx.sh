
checkpoint_file="checkpoints/checkpoint-19160_10-best-val-loss-epoch=227-step=684.ckpt"
onnx_file="checkpoints/checkpoint-19160_10-best-val-loss-epoch=227-step=684.onnx"
model_args_file="checkpoints/args-19160_10-best-val-loss-epoch=227-step=684.json"
pixel_spacing="0.2"
crop_size="512"

device="cpu"  # or cuda

# left hip is right on image, stored in dcm_RasL.pts
python -u export_seg_to_onnx.py \
  --checkpoint "${checkpoint_file}" \
  --model-args "${model_args_file}" \
  --crop-size "${crop_size}" \
  --output-onnx "${onnx_file}"
