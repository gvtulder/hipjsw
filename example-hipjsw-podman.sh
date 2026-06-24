output_dir="outputs-podman"
mkdir -p "${output_dir}"

echo podman build -t hipjsw .

# process csv
podman run --rm -it \
  --volume .:/workdir \
  hipjsw \
  --input-csv "example-input.csv" \
  --output-plots "${output_dir}/{scan_id}.png" \
  --output-trace "${output_dir}/{scan_id}.npz" \
  --output-csv "${output_dir}/test-multi.csv" \
  --print-json
