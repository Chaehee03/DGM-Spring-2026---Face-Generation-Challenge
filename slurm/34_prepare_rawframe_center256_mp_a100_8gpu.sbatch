#!/bin/bash
#SBATCH --job-name=dgm_prep_raw_mp
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
export PYTHONUNBUFFERED=1

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face

SRC=/data/chsong/dgm-face/frames_raw_3f_0205080
DST=/data/chsong/dgm-face/frames_raw_3f_0205080_center256_jpg

echo "Start: $(date)"
echo "SRC=$SRC"
echo "DST=$DST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python scripts/prepare_rawframe_center256_mp.py \
  --src "$SRC" \
  --dst "$DST" \
  --workers 32 \
  --size 256 \
  --quality 95

echo "Done: $(date)"
