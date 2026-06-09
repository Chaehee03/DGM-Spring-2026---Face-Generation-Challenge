#!/bin/bash
#SBATCH --job-name=dgm_sg2_i3g
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-02
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
export PYTHONUNBUFFERED=1

source /workspace/chsong/miniconda3/bin/activate stylegan

cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

DATA=/data/chsong/dgm-face/datasets/celebvhq_256_insight_3f_std.zip
OUTDIR=/data/chsong/dgm-face/training-runs
PRETRAIN=/workspace/chsong/dgm-face/stylegan2-ada-pytorch/pretrained/ffhq256.pkl

mkdir -p "$OUTDIR"

LATEST=$(find "$OUTDIR" -name "network-snapshot-*.pkl" | sort | tail -n 1)

if [ -n "$LATEST" ]; then
  RESUME="$LATEST"
  echo "Resume from latest snapshot: $RESUME"
else
  RESUME="$PRETRAIN"
  echo "Resume from pretrained FFHQ: $RESUME"
fi

echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATA=$DATA"
echo "OUTDIR=$OUTDIR"
echo "RESUME=$RESUME"

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
  --gpus=3 \
  --batch=96 \
  --gamma=2 \
  --mirror=1 \
  --aug=ada \
  --target=0.6 \
  --snap=10 \
  --kimg=3000 \
  --metrics=fid50k_full

echo "Done: $(date)"
