#!/bin/bash
#SBATCH --job-name=dgm_sub_080
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face

python tools/make_submission_from_candidates.py \
  --base /data/chsong/dgm-face/generated_candidates_insight_sweep \
  --outroot /data/chsong/dgm-face/submissions/pure080_v2 \
  --spec "0.80:psi_0_80:1000" \
  --seed 2027
