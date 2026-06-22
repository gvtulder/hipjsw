
segmentation_model="checkpoints/checkpoint-19160_10-best-val-loss-epoch=227-step=684.onnx"
pixel_spacing="0.2"
crop_size="512"

# left hip is right on image, stored in dcm_RasL.pts
python -u predictor.py \
  --segmentation-model "${segmentation_model}" \
  --input-dicom "../images/OAI-9763898-V00-20051017.dcm" \
  --input-points "../images/OAI-9763898-V00-20051017.dcm_L.pts" \
  --side right \
  --pixel-spacing "${pixel_spacing}" \
  --crop-size "${crop_size}" \
  --save-image "test-{side}-predict-img.png" \
  --save-segmentation "test-{side}-predict-seg.png" \
