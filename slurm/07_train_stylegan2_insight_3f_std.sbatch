#!/bin/bash
#SBATCH --job-name=dgm_sg2_i3f
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

export PYTHONUNBUFFERED=1
source /workspace/chsong/miniconda3/bin/activate stylegan

cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
nvidia-smi

python train.py \
  --outdir=/data/chsong/dgm-face/training-runs \
  --data=/data/chsong/dgm-face/datasets/celebvhq_256_insight_3f_std.zip \
  --resume=pretrained/ffhq256.pkl \
  --cfg=paper256 \
  --gpus=1 \
  --batch=32 \
  --gamma=2 \
  --mirror=1 \
  --aug=ada \
  --target=0.6 \
  --snap=10 \
  --kimg=3000 \
  --metrics=fid50k_full

echo "Done: $(date)"
