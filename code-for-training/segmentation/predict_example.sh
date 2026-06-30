#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output slurm-logs/%x-%A_%a.out
#SBATCH --error slurm-logs/%x-%A_%a.err
#SBATCH --array=4

source data/gijs/venv/bin/activate


define_parameters() {

  pick dataset     v6-20260324 \
                   v6-20260303-FB \
                   v6-20260331-TEST-SEG \
                   testset_jsw_fleur
  pick resolution  0.2mm-512x512
  pick split       seed123

  pick checkpoint \
    19160_10

}

. parameter_helper.sh


if [[ $dataset == v6-20260331-TEST-SEG ]] ; then
  hdf5="manual-test-for-hip-segmentation-v6-20260331-0.2mm-512x512.h5"
  subset="eval"
  list="lists/testset_segment_20260318.txt"
elif [[ $dataset == testset_jsw_fleur ]] ; then
  hdf5="only-EVAL-for-hip-segmentation-v5-20260126-0.2mm-512x512.h5"
  subset="eval"
  list="lists/testset_jsw_fleur.txt"
else
  hdf5="manual-for-hip-segmentation-${dataset}-${resolution}.h5"
  subset="val"
  list="lists/split.20260324.fiveval.txt"
fi

checkpoint_file=$( ls -1 -t checkpoints/${checkpoint}/last.ckpt | head -n 1 )
output="pred-for-comparison-20260414/${checkpoint}.${dataset}.predictions.last.${subset}.ckpt.h5"

checkpoint_file=$( ls -1 -t checkpoints/${checkpoint}/best-val-loss* | head -n 1 )
output="pred-for-comparison-20260414/${checkpoint}.${dataset}.predictions.best-val-loss.${subset}.ckpt.h5"

if [[ -f "${output}" ]] ; then
  echo "Output already exists: ${output}."
  exit
fi

mkdir -p $(dirname "$output")

OMP_NUM_THREADS=1 \
MPI_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
HDF5_USE_FILE_LOCKING=FALSE \
python -u predict.py \
  --checkpoint "${checkpoint_file}" \
  --model-args "checkpoints/${checkpoint}/args.json" \
  --allow-missing-samples \
  --data "${hdf5}" \
  --data-split "${list}" \
  --predict-subset ${subset} \
  --output "${output}" \
  --device=cuda
# --data-split "lists/small-demo-subset.txt" \
# --output "checkpoints/${checkpoint}/predictions.last.small-demo-subset.ckpt.h5" \

