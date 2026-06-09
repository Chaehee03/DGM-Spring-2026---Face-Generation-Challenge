#!/bin/bash
#SBATCH --job-name=dgm_ext3f
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

export PYTHONUNBUFFERED=1
source /workspace/chsong/miniconda3/bin/activate stylegan

cd /workspace/chsong/dgm-face

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

echo "MP4 count:"
find /data/chsong/dgm-face/videos -name "*.mp4" | wc -l

python scripts/extract_frames.py \
  --video-dir /data/chsong/dgm-face/videos \
  --out-dir /data/chsong/dgm-face/frames_raw_3f_0205080 \
  --meta-path /data/chsong/dgm-face/metadata/extracted_frames_3f_0205080.csv \
  --fracs 0.2,0.5,0.8 \
  --workers 8 \
  --jpeg-quality 95

echo "Raw frames:"
find /data/chsong/dgm-face/frames_raw_3f_0205080 -name "*.jpg" | wc -l

echo "Metadata:"
ls -lh /data/chsong/dgm-face/metadata/extracted_frames_3f_0205080.csv

echo "Done: $(date)"
