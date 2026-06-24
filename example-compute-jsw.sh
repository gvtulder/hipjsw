
segmentation_model="src/hipjsw/checkpoints/checkpoint-19160_10-best-val-loss-epoch=227-step=684.onnx"
pixel_spacing="0.2"
crop_size="512"

output_dir="outputs/"

mkdir -p "${output_dir}"

# left hip is right on image, stored in dcm_RasL.pts
python -u src/hipjsw \
  --segmentation-model "${segmentation_model}" \
  --input-image "../images/OAI-9763898-V00-20051017.dcm" \
  --input-points "../images/OAI-9763898-V00-20051017.dcm_L.pts" \
  --side right \
  --scan-id "OAI-976389/V00/right" \
  --pixel-spacing "${pixel_spacing}" \
  --crop-size "${crop_size}" \
  --plot-types overview \
  --output-plots "${output_dir}/{scan_id}.png" \
  --output-trace "${output_dir}/{scan_id}.npz" \
  --output-csv "${output_dir}/test-single.csv" \
  --print-json \


# process csv
python -u src/hipjsw \
  --segmentation-model "${segmentation_model}" \
  --input-csv "example-input.csv" \
  --pixel-spacing "${pixel_spacing}" \
  --crop-size "${crop_size}" \
  --plot-types overview \
  --output-plots "${output_dir}/{scan_id}.png" \
  --output-trace "${output_dir}/{scan_id}.npz" \
  --output-csv "${output_dir}/test-multi.csv" \
  --print-json \


# process csv without points
python -u src/hipjsw \
  --segmentation-model "${segmentation_model}" \
  --input-csv "example-input-without-points.csv" \
  --pixel-spacing "${pixel_spacing}" \
  --crop-size "${crop_size}" \
  --plot-types overview \
  --output-plots "${output_dir}/{scan_id}.png" \
  --output-trace "${output_dir}/{scan_id}.npz" \
  --output-csv "${output_dir}/test-multi-detect-hips.csv" \
  --print-json \


# single image
python -u src/hipjsw \
  --input-image "../images/OAI-9763898-V00-20051017.dcm" \
  --output-plots "${output_dir}/test-simple-{side}.png" \
  --output-trace "${output_dir}/test-simple-{side}.npz" \
  --output-csv "${output_dir}/test-simple.csv" \
  --print-json


# single image
python -u src/hipjsw \
  --input-image "../images/utah-edu-collections.jpg" \
  --output-plots "${output_dir}/test-jpg-{side}.png" \
  --output-trace "${output_dir}/test-jpg-{side}.npz" \
  --output-csv "${output_dir}/test-jpg.csv" \
  --print-json


# single image
python -u src/hipjsw \
  --input-image "../images/Medical_X-Ray_imaging_SAL07_nevit.jpg" \
  --output-plots "${output_dir}/test-jpg2-{side}.png" \
  --output-trace "${output_dir}/test-jpg2-{side}.npz" \
  --output-csv "${output_dir}/test-jpg2.csv" \
  --print-json


# single image
python -u src/hipjsw \
  --input-image "../images/Medical_X-Ray_imaging_SAL07_nevit.jpg" \
  --output-plots "${output_dir}/test-jpg2-{side}.png" \
  --output-trace "${output_dir}/test-jpg2-{side}.npz" \
  --output-csv "${output_dir}/test-jpg2.csv" \
  --plot-left-right \
  --print-json


