#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output slurm-logs/%x-%A_%a.out
#SBATCH --error slurm-logs/%x-%A_%a.err
#SBATCH --array=1

source venv/bin/activate


define_parameters() {

  pick max_epochs          200
  pick lr                  0.001
  pick mb_size             32

  pick model               HipSegmentation_UNet_4_Wide8
  pick dataset             v5-20251215
  pick resolution          0.2mm-512x512
  pick split               exmanualtraintest.seed123

}

. parameter_helper.sh

hdf5="all-for-hip-segmentation-${dataset}-${resolution}.h5"

OMP_NUM_THREADS=1 \
MPI_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
HDF5_USE_FILE_LOCKING=FALSE \
python -u train.py \
  --experiment-name "segmentation-${model}-${dataset}-${split}-${resolution}-gamma-elasticstrong-lr-${lr}" \
  --dataset SegmentationDataset \
  --model $model \
  --data-augment intensity3 gamma elastic-strong \
  --data "$hdf5" \
  --data-split "lists/split.20251215.${split}.txt" \
  --allow-missing-samples \
  --num-classes 5 \
  --num-workers 10 \
  --max-epochs $max_epochs \
  --lr $lr \
  --mb-size $mb_size \
  --seed 123 \
  --log-dir "logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  --checkpoint-dir "checkpoints/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
