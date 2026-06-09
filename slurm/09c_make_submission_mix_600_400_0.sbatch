#!/bin/bash
#SBATCH --job-name=dgm_sub_640
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
export PYTHONUNBUFFERED=1

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face

python tools/make_submission_from_candidates.py \
  --base /data/chsong/dgm-face/generated_candidates_insight_sweep \
  --outroot /data/chsong/dgm-face/submissions/mix_600_400_0_v3 \
  --spec "0.65:psi_0_65:600,0.80:psi_0_80:400" \
  --seed 2028
