#!/bin/bash
#SBATCH --job-name=dgm_clean2k
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

DATA=/data/chsong/dgm-face/datasets/celebvhq_256_insight_3f_std.zip
PRETRAIN=/workspace/chsong/dgm-face/stylegan2-ada-pytorch/pretrained/ffhq256.pkl
OUTDIR=/data/chsong/dgm-face/training-runs-ffhq-ada-color-clean-k2000

mkdir -p "$OUTDIR"

echo "Start MAIN: $(date)"
echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATA=$DATA"
echo "PRETRAIN=$PRETRAIN"
echo "OUTDIR=$OUTDIR"
nvidia-smi

python train.py \
  --outdir="$OUTDIR" \
  --data="$DATA" \
  --resume="$PRETRAIN" \
  --cfg=paper256 \
  --gpus=8 \
  --batch=128 \
  --gamma=2 \
  --mirror=1 \
  --aug=ada \
  --augpipe=color \
  --target=0.6 \
  --snap=25 \
  --kimg=2000 \
  --metrics=none \
  --workers=8

echo "Done MAIN: $(date)"
