#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output slurm-logs/%x-%A_%a.out
#SBATCH --error slurm-logs/%x-%A_%a.err
#SBATCH --array=1

source venv/bin/activate


define_parameters() {

  pick max_epochs          300
  pick mb_size             32

  pick model               HipSegmentation_UNet_4_Wide8
  pick resolution          0.2mm-512x512
  pick split               20260303-FB.fiveval

  pick augment             elastic-strong

  pick pretrained          18915_1

  pick lr                  0.0001

}

. parameter_helper.sh

hdf5="manual-for-hip-segmentation-${dataset}-${resolution}.h5"

checkpoint=$( ls -1 -t checkpoints/${pretrained}/best-val-loss* | head -n 1 )

if [[ $augment == noaugment ]] ; then
  augment_options=""
else
  augment_options="--data-augment intensity3 gamma ${augment}"
fi

if [[ $split == 20260324.fiveval ]] ; then
  dataset=v6-20260324
elif [[ $split == 20260303-FB.fiveval ]] ; then
  dataset=v6-20260303-FB
else
  dataset=v6-20260303
fi
hdf5="manual-for-hip-segmentation-${dataset}-${resolution}.h5"
datalist="lists/split.${split}.txt"

if [[ $split == 20260303-FB.fiveval ]] ; then
  datalist="lists/split.20260303.fiveval.txt"
fi

OMP_NUM_THREADS=1 \
MPI_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
HDF5_USE_FILE_LOCKING=FALSE \
python -u train.py \
  --experiment-name "segmentation-${model}-refine-${pretrained}-MANUAL-${dataset}-${split}-${resolution}-gamma-${augment}-lr-${lr}" \
  --dataset SegmentationDataset \
  --model $model \
  ${augment_options} \
  --data "$hdf5" \
  --data-split "${datalist}" \
  --allow-missing-samples \
  --num-classes 5 \
  --num-workers 10 \
  --max-epochs $max_epochs \
  --lr $lr \
  --mb-size $mb_size \
  --seed 123 \
  --log-dir "logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  --checkpoint-dir "checkpoints/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  --pretrained-checkpoint "${checkpoint}"
