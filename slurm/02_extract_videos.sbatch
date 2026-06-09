#!/bin/bash
#SBATCH --job-name=dgm_untar
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e

TAR=/data/chsong/dgm-face/celebvhq_hf/videos.tar
OUT=/data/chsong/dgm-face/videos

mkdir -p "$OUT"

echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

echo "Checking tar..."
ls -lh "$TAR"
echo "Tar mp4 count:"
tar -tf "$TAR" | grep -E '\.mp4$' | wc -l

echo "Currently extracted mp4 count:"
find "$OUT" -name "*.mp4" | wc -l

echo "Extracting..."
tar -xf "$TAR" -C "$OUT"

echo "Done: $(date)"
echo "Extracted mp4 count:"
find "$OUT" -name "*.mp4" | wc -l

echo "Output size:"
du -sh "$OUT"
