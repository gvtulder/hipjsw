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

VERSION="20260303-FB"
RESOLUTION="0.2"
CROP="512"

# scans with manual annotations
INPUT_CSV="data-ro/JSAnnot/manual-annot-20260303-GvT_FB.csv"
XRAYS_PATH="data-ro/Baseline/XRAYS/"
POINTS_PATH="data-ro/Baseline/PTSFILES/"
MANUAL_PATH="data-ro/JSAnnot/Output_GvT_FB/"
OUTPUT_FILE="manual-for-hip-segmentation-v6-${VERSION}-${RESOLUTION}mm-${CROP}x${CROP}.h5"

mkdir -p data
HDF5_USE_FILE_LOCKING=FALSE \
python convert_for_segmentation_v6_manual.py \
  --skip-missing-files \
  --input-csv "${INPUT_CSV}" \
  --images-path "${XRAYS_PATH}" \
  --points-path "${POINTS_PATH}" \
  --manual-path "${MANUAL_PATH}" \
  --output "${OUTPUT_FILE}" \
  --target-pixel-spacing "${RESOLUTION}" \
  --crop-size "${CROP}" \
  --num-workers 8 | \
  tee "${OUTPUT_FILE}.errors.txt"

