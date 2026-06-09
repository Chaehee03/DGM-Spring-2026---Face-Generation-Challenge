#!/bin/bash
#SBATCH --job-name=dgm_ada_c8_300
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

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

DATA=/data/chsong/dgm-face/datasets/celebvhq_256_insight_3f_std.zip
OUTDIR=/data/chsong/dgm-face/training-runs-ada-color-from300

mkdir -p "$OUTDIR"

RESUME=$(python - <<'PY'
from pathlib import Path
import re

root = Path("/data/chsong/dgm-face/training-runs")
target = 300

cands = []
for p in root.rglob("network-snapshot-*.pkl"):
    m = re.search(r"network-snapshot-(\d+)\.pkl$", p.name)
    if m:
        k = int(m.group(1))
        cands.append((abs(k-target), k, p))

if not cands:
    print("/workspace/chsong/dgm-face/stylegan2-ada-pytorch/pretrained/ffhq256.pkl")
else:
    cands.sort()
    print(cands[0][2])
PY
)

echo "Start: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATA=$DATA"
echo "RESUME=$RESUME"
echo "OUTDIR=$OUTDIR"
nvidia-smi

python train.py \
  --outdir="$OUTDIR" \
  --data="$DATA" \
  --resume="$RESUME" \
  --cfg=paper256 \
  --gpus=8 \
  --batch=64 \
  --gamma=2 \
  --mirror=1 \
  --aug=ada \
  --augpipe=color \
  --target=0.6 \
  --snap=25 \
  --kimg=1200 \
  --metrics=none

echo "Done: $(date)"
