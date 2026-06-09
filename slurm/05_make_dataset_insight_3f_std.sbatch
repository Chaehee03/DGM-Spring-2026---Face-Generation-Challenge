#!/bin/bash
#SBATCH --job-name=dgm_mkdata_i
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-08:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

export PYTHONUNBUFFERED=1
source /workspace/chsong/miniconda3/bin/activate stylegan

cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

SRC=/data/chsong/dgm-face/faces_256_insight_3f_std
DST=/data/chsong/dgm-face/datasets/celebvhq_256_insight_3f_std.zip

mkdir -p /data/chsong/dgm-face/datasets

echo "Aligned face count:"
find "$SRC" -name "*.png" | wc -l

python dataset_tool.py \
  --source="$SRC" \
  --dest="$DST" \
  --resolution=256x256

ls -lh "$DST"
