#!/bin/bash
#SBATCH --job-name=dgm_verify_vid
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
VIDEO_DIR=/data/chsong/dgm-face/videos
BAD_LIST=/data/chsong/dgm-face/metadata/bad_videos.txt

mkdir -p /data/chsong/dgm-face/metadata
: > $BAD_LIST

echo "Before:"
find $VIDEO_DIR -name "*.mp4" | wc -l

find $VIDEO_DIR -name "*.mp4" | while read f; do
    ffprobe -v error -i "$f" -show_format > /dev/null 2>&1 || echo "$f" >> $BAD_LIST
done

echo "Bad files:"
wc -l $BAD_LIST

while read f; do
    [ -n "$f" ] && rm -f "$f"
done < $BAD_LIST

echo "After:"
find $VIDEO_DIR -name "*.mp4" | wc -l
