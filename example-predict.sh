
checkpoint_file="checkpoints/checkpoint-19160_10-best-val-loss-epoch=227-step=684.ckpt"
model_args_file="checkpoints/args-19160_10-best-val-loss-epoch=227-step=684.json"
pixel_spacing="0.2"
crop_size="512"

device="cpu"  # or cuda

# left hip is right on image, stored in dcm_RasL.pts
python -u predictor.py \
  --device="${device}" \
  --checkpoint "${checkpoint_file}" \
  --model-args "${model_args_file}" \
  --input-dicom "../images/OAI-9763898-V00-20051017.dcm" \
  --input-points "../images/OAI-9763898-V00-20051017.dcm_L.pts" \
  --side right \
  --pixel-spacing "${pixel_spacing}" \
  --crop-size "${crop_size}" \
  --save-image "test-{side}-predict-img.png" \
  --save-segmentation "test-{side}-predict-seg.png" \
