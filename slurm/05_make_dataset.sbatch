#!/bin/bash
#SBATCH --job-name=dgm_mkdata
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

export PYTHONUNBUFFERED=1
source /workspace/chsong/miniconda3/bin/activate stylegan

cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

python dataset_tool.py \
  --source=/data/chsong/dgm-face/faces_256_ffhq \
  --dest=/data/chsong/dgm-face/datasets/celebvhq_256_ffhqalign.zip \
  --resolution=256x256

ls -lh /data/chsong/dgm-face/datasets/celebvhq_256_ffhqalign.zip
