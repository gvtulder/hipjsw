#!/bin/bash
# list all images from OAI and CHECK with pointfiles,
# create images and label files for YOLO training
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output slurm-logs/%x-%A_%a.out
#SBATCH --error slurm-logs/%x-%A_%a.err

source venv/bin/activate

VERSION="20260119"
RESOLUTION="0.2"

# CSV with improved TASOAC data: only one image per subject, TASOAC-00000
INPUT_CSV_A="data-src/Baseline/20251022_avail_img_in_dir_Baseline--MODIFIED-Gijs.csv"
INPUT_CSV_B="data-src/Follow-up/20251029_avail_img_in_dir_Follow-up_ERGO5.csv"
XRAYS_PATH="data-ro/Baseline/XRAYS/"
POINTS_PATH="data-ro/Baseline/PTSFILES/"
OUTPUT_FILE="data-for-yolo"

rm -r data-for-yolo/images
rm -r data-for-yolo/labels

mkdir -p data
HDF5_USE_FILE_LOCKING=FALSE \
python convert_for_yolo.py \
  --skip-missing-files \
  --input-csv "${INPUT_CSV_A}" "${INPUT_CSV_B}" \
  --subsets "../lists/split.20251215.manual.seed123.txt" \
  --images-path "${XRAYS_PATH}" \
  --points-path "${POINTS_PATH}" \
  --output-path "${OUTPUT_FILE}" \
  --target-pixel-spacing "${RESOLUTION}" \
  --min-aspect-ratio 0.66 \
  --num-workers 12 | \
  tee "${OUTPUT_FILE}.errors.txt"

