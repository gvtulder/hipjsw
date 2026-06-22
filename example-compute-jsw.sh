
checkpoint_file="checkpoints/checkpoint-19160_10-best-val-loss-epoch=227-step=684.ckpt"
model_args_file="checkpoints/args-19160_10-best-val-loss-epoch=227-step=684.json"
pixel_spacing="0.2"
crop_size="512"

device="cpu"  # or cuda

output_dir="outputs/"

mkdir -p "${output_dir}"

# left hip is right on image, stored in dcm_RasL.pts
python -u compute_jsw.py \
  --device="${device}" \
  --checkpoint "${checkpoint_file}" \
  --model-args "${model_args_file}" \
  --input-dicom "../images/OAI-9763898-V00-20051017.dcm" \
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
python -u compute_jsw.py \
  --device="${device}" \
  --checkpoint "${checkpoint_file}" \
  --model-args "${model_args_file}" \
  --input-csv "example-input.csv" \
  --pixel-spacing "${pixel_spacing}" \
  --crop-size "${crop_size}" \
  --plot-types overview \
  --output-plots "${output_dir}/{scan_id}.png" \
  --output-trace "${output_dir}/{scan_id}.npz" \
  --output-csv "${output_dir}/test-multi.csv" \
  --print-json \


# process csv without points
python -u compute_jsw.py \
  --device="${device}" \
  --checkpoint "${checkpoint_file}" \
  --model-args "${model_args_file}" \
  --input-csv "example-input-without-points.csv" \
  --pixel-spacing "${pixel_spacing}" \
  --crop-size "${crop_size}" \
  --plot-types overview \
  --output-plots "${output_dir}/{scan_id}.png" \
  --output-trace "${output_dir}/{scan_id}.npz" \
  --output-csv "${output_dir}/test-multi-detect-hips.csv" \
  --print-json \


