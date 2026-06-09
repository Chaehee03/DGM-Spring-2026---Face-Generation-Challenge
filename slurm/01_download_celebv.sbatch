#!/bin/bash
#SBATCH --job-name=dgm_download
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

export PYTHONUNBUFFERED=1
source /workspace/chsong/miniconda3/bin/activate stylegan

cd /workspace/chsong/dgm-face/CelebV-HQ

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

yt-dlp --version || true
ffmpeg -version | head -n 2 || true

# 실제 repo 스크립트명을 먼저 확인
ls

# 아래는 후보. 실제 파일명에 맞게 하나만 남겨서 실행.
if [ -f download_and_process.py ]; then
    python download_and_process.py
elif [ -f download_celebvhq.py ]; then
    python download_celebvhq.py --output_dir /data/chsong/dgm-face/videos
else
    echo "No known download script found. Check repository files."
    exit 1
fi

echo "Done: $(date)"
