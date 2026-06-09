#!/bin/bash
#SBATCH --job-name=dgm_gen_sw
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-06:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%A_%a.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%A_%a.err

set -e
export PYTHONUNBUFFERED=1

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

SNAP=/data/chsong/dgm-face/training-runs/00010-celebvhq_256_insight_3f_std-mirror-paper256-gamma2-kimg3000-batch64-noaug-resumecustom/network-snapshot-003000.pkl
BASE_OUT=/data/chsong/dgm-face/generated_candidates_insight_sweep

# task mapping:
# 0,1 -> psi 0.65 (seeds 0-4999, 5000-9999)
# 2,3 -> psi 0.80 (seeds 0-4999, 5000-9999)
# 4,5 -> psi 1.00 (seeds 0-4999, 5000-9999)

TASK=${SLURM_ARRAY_TASK_ID}
PSI_GROUP=$(( TASK / 2 ))
SHARD=$(( TASK % 2 ))

if [ "$PSI_GROUP" -eq 0 ]; then
  PSI="0.65"
elif [ "$PSI_GROUP" -eq 1 ]; then
  PSI="0.80"
else
  PSI="1.00"
fi

if [ "$SHARD" -eq 0 ]; then
  SEEDS="0-4999"
  SHARD_NAME="part0"
else
  SEEDS="5000-9999"
  SHARD_NAME="part1"
fi

# directory-safe psi string
PSI_TAG=$(echo "$PSI" | tr '.' '_')

OUTDIR=${BASE_OUT}/psi_${PSI_TAG}/${SHARD_NAME}
mkdir -p "$OUTDIR"

echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SNAP=$SNAP"
echo "PSI=$PSI"
echo "SEEDS=$SEEDS"
echo "OUTDIR=$OUTDIR"

nvidia-smi
test -f "$SNAP"

python generate.py \
  --network="$SNAP" \
  --outdir="$OUTDIR" \
  --seeds="$SEEDS" \
  --trunc="$PSI"

echo "Generated count:"
find "$OUTDIR" -maxdepth 1 -type f -name "*.png" | wc -l
echo "Done: $(date)"
