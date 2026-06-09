#!/bin/bash
#SBATCH --job-name=dgm_cont2k
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

TARGET_TOTAL_KIMG=${TARGET_TOTAL_KIMG:-2000}

if [ -z "$PREV_JOB" ]; then
  echo "ERROR: PREV_JOB is not set."
  exit 1
fi

echo "Start CONT checker: $(date)"
echo "Current job: $SLURM_JOB_ID"
echo "Previous job: $PREV_JOB"
echo "Target total kimg: $TARGET_TOTAL_KIMG"

PREV_STATE=$(sacct -j "$PREV_JOB" -X -n -o State | head -n 1 | awk '{print $1}')
echo "Previous job state: $PREV_STATE"

if [ "$PREV_STATE" != "TIMEOUT" ]; then
  echo "Previous job did not TIMEOUT. No continuation training."
  exit 0
fi

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

DATA=/data/chsong/dgm-face/datasets/celebvhq_256_insight_3f_std.zip
MAIN_ROOT=/data/chsong/dgm-face/training-runs-ffhq-ada-color-clean-k2000
CONT_ROOT=/data/chsong/dgm-face/training-runs-ffhq-ada-color-clean-k2000-cont

mkdir -p "$CONT_ROOT"

read RESUME LATEST_KIMG REMAINING_KIMG < <(TARGET_TOTAL_KIMG="$TARGET_TOTAL_KIMG" python - <<'PY'
from pathlib import Path
import os
import re

target = int(os.environ["TARGET_TOTAL_KIMG"])
root = Path("/data/chsong/dgm-face/training-runs-ffhq-ada-color-clean-k2000")

cands = []
for p in root.rglob("network-snapshot-*.pkl"):
    m = re.search(r"network-snapshot-(\d+)\.pkl$", p.name)
    if m:
        k = int(m.group(1))
        cands.append((k, p.stat().st_mtime, p))

if not cands:
    raise SystemExit("No MAIN snapshot found to resume from.")

cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
latest_kimg, _, latest_path = cands[0]
remaining = max(target - latest_kimg, 0)

print(latest_path, latest_kimg, remaining)
PY
)

echo "Latest MAIN snapshot: $RESUME"
echo "Latest MAIN kimg: $LATEST_KIMG"
echo "Remaining kimg to reach $TARGET_TOTAL_KIMG: $REMAINING_KIMG"

if [ "$REMAINING_KIMG" -le 0 ]; then
  echo "Already reached target. No continuation training."
  exit 0
fi

echo "Start CONT training: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATA=$DATA"
echo "RESUME=$RESUME"
echo "CONT_ROOT=$CONT_ROOT"
nvidia-smi

python train.py \
  --outdir="$CONT_ROOT" \
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
  --kimg="$REMAINING_KIMG" \
  --metrics=none \
  --workers=8

echo "Done CONT: $(date)"
