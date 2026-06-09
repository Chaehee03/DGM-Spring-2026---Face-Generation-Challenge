#!/bin/bash
#SBATCH --job-name=dgm_raw3f_5k5
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

DATA=/data/chsong/dgm-face/datasets/celebvhq_rawframe_3f_0205080_center256_jpg.zip

RESUME=/data/chsong/dgm-face/training-runs-rawframe-ffhq-ada-color-k2000/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg2000-batch128-ada-target0.6-color-resumecustom/network-snapshot-002000.pkl

OUTDIR=/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k5500-from2000

# 2000 -> 5500 이므로 추가 3500 kimg
ADDITIONAL_KIMG=3500

mkdir -p "$OUTDIR"

echo "Start RAWFRAME 3F continuation: $(date)"
echo "DATA=$DATA"
echo "RESUME=$RESUME"
echo "OUTDIR=$OUTDIR"
echo "ADDITIONAL_KIMG=$ADDITIONAL_KIMG"
echo "Target effective total kimg = 2000 + 3500 = 5500"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

test -f "$DATA"
test -f "$RESUME"

nvidia-smi

python train.py \
  --outdir="$OUTDIR" \
  --data="$DATA" \
  --resume="$RESUME" \
  --cfg=paper256 \
  --gpus=8 \
  --batch=128 \
  --gamma=2 \
  --mirror=1 \
  --aug=ada \
  --augpipe=color \
  --target=0.6 \
  --snap=25 \
  --kimg="$ADDITIONAL_KIMG" \
  --metrics=none \
  --workers=8

echo "Done RAWFRAME 3F continuation: $(date)"
