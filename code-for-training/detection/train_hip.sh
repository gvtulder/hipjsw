#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output slurm-logs/%x-%A_%a.out
#SBATCH --error slurm-logs/%x-%A_%a.err

source venv/bin/activate

python tools/train.py --model configs/models/edge_s.yaml --data dataset.yaml --epochs 100 --img_size 640 --batch_size 8 --workers 4

