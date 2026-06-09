#!/bin/bash
#SBATCH --job-name=dgm_prep_raw
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%A_%a.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%A_%a.err

set -e
export PYTHONUNBUFFERED=1

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face

SRC=/data/chsong/dgm-face/frames_raw_3f_0205080
DST=/data/chsong/dgm-face/frames_raw_3f_0205080_center256_jpg

python scripts/prepare_rawframe_center256.py \
  --src "$SRC" \
  --dst "$DST" \
  --shard "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --size 256 \
  --quality 95

echo "Done: $(date)"
