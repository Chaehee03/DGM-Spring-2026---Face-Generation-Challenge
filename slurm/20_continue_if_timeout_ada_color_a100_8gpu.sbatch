#!/bin/bash
#SBATCH --job-name=dgm_ada_cont
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

if [ -z "$PREV_JOB" ]; then
  echo "ERROR: PREV_JOB is not set."
  exit 1
fi

echo "Current job: $SLURM_JOB_ID"
echo "Previous job: $PREV_JOB"

PREV_STATE=$(sacct -j "$PREV_JOB" -X -n -o State | head -n 1 | awk '{print $1}')
echo "Previous job state: $PREV_STATE"

if [ "$PREV_STATE" != "TIMEOUT" ]; then
  echo "Previous job did not TIMEOUT. No continuation needed."
  echo "Exit without training."
  exit 0
fi

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

DATA=/data/chsong/dgm-face/datasets/celebvhq_256_insight_3f_std.zip
BASE_ROOT=/data/chsong/dgm-face
OUTDIR=/data/chsong/dgm-face/training-runs-ffhq-ada-color-timeout-cont

mkdir -p "$OUTDIR"

RESUME=$(python - <<'PY'
from pathlib import Path

roots = [
    Path("/data/chsong/dgm-face/training-runs-ffhq-ada-color-k2000"),
    Path("/data/chsong/dgm-face/training-runs-ffhq-ada-color-timeout-cont"),
]

cands = []
for root in roots:
    if root.exists():
        for p in root.rglob("network-snapshot-*.pkl"):
            cands.append((p.stat().st_mtime, p))

if not cands:
    raise SystemExit("No snapshot found to resume from.")

cands.sort(reverse=True)
print(cands[0][1])
PY
)

echo "Start continuation: $(date)"
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
  --batch=128 \
  --gamma=2 \
  --mirror=1 \
  --aug=ada \
  --augpipe=color \
  --target=0.6 \
  --snap=25 \
  --kimg=500 \
  --metrics=none \
  --workers=8

echo "Done continuation: $(date)"
