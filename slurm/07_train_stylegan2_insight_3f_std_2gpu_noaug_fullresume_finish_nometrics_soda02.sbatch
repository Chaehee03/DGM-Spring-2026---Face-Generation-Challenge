#!/bin/bash
#SBATCH --job-name=dgm_sg2_i2g_full
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-02
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=TERM@600
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
export PYTHONUNBUFFERED=1
export STYLEGAN_FULL_STATE=/data/chsong/dgm-face/training-runs/full_state_noaug_latest.pt

source /workspace/chsong/miniconda3/bin/activate stylegan

cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

DATA=/data/chsong/dgm-face/datasets/celebvhq_256_insight_3f_std.zip
OUTDIR=/data/chsong/dgm-face/training-runs
PRETRAIN=/workspace/chsong/dgm-face/stylegan2-ada-pytorch/pretrained/ffhq256.pkl

mkdir -p "$OUTDIR"

RESUME="$PRETRAIN"

if [ -f "$STYLEGAN_FULL_STATE" ]; then
  echo "Full state exists: $STYLEGAN_FULL_STATE"
  echo "training_loop.py will restore full training state."
else
  echo "No full state found. Starting from FFHQ pretrained."
fi

echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATA=$DATA"
echo "OUTDIR=$OUTDIR"
echo "RESUME=$RESUME"
echo "STYLEGAN_FULL_STATE=$STYLEGAN_FULL_STATE"

echo "=== nvidia-smi ==="
nvidia-smi

echo "=== torch cuda check ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

echo "=== file checks ==="
test -f "$DATA"
test -f "$RESUME"

echo "=== start training ==="

python train.py \
  --outdir="$OUTDIR" \
  --data="$DATA" \
  --resume="$RESUME" \
  --cfg=paper256 \
  --gpus=2 \
  --batch=64 \
  --gamma=2 \
  --mirror=1 \
  --aug=noaug \
  --snap=1 \
  --kimg=3000 \
  --metrics=none

echo "Done: $(date)"
