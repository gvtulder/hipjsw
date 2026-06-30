#!/bin/bash
# list all images from OAI and CHECK with pointfiles,
# create HDF5 files for a hip segmentation
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output slurm-logs/%x-%A_%a.out
#SBATCH --error slurm-logs/%x-%A_%a.err

source venv/bin/activate

VERSION="20251215"
RESOLUTION="0.2"
CROP="512"

INPUT_CSV_A="data-src/Baseline/20251022_avail_img_in_dir_Baseline.csv"
INPUT_CSV_B="data-src/Follow-up/20251029_avail_img_in_dir_Follow-up_ERGO5.csv"
XRAYS_PATH="data-ro/Baseline/XRAYS/"
POINTS_PATH="data-ro/Baseline/PTSFILES/"
OUTPUT_FILE="all-for-hip-segmentation-v5-${VERSION}-${RESOLUTION}mm-${CROP}x${CROP}.h5"

mkdir -p data
HDF5_USE_FILE_LOCKING=FALSE \
python convert_for_segmentation_v5.py \
  --skip-missing-files \
  --input-csv "${INPUT_CSV_A}" "${INPUT_CSV_B}" \
  --images-path "${XRAYS_PATH}" \
  --points-path "${POINTS_PATH}" \
  --output "${OUTPUT_FILE}" \
  --target-pixel-spacing "${RESOLUTION}" \
  --crop-size "${CROP}" \
  --num-workers 8 | \
  tee "${OUTPUT_FILE}.errors.txt"

