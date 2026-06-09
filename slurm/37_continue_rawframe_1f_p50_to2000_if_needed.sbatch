#!/bin/bash
#SBATCH --job-name=dgm_1fp50_cont
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

MAIN_JOB=${MAIN_JOB:?MAIN_JOB is not set}

DATA=/data/chsong/dgm-face/datasets/celebvhq_rawframe_1f_p50_center256_jpg.zip
OUTROOT=/data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000
CONT_OUTROOT=/data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000-cont
TARGET_KIMG=2000

mkdir -p "$CONT_OUTROOT"

echo "Start RAWFRAME 1F P50 continuation: $(date)"
echo "MAIN_JOB=$MAIN_JOB"
echo "DATA=$DATA"
echo "OUTROOT=$OUTROOT"
echo "CONT_OUTROOT=$CONT_OUTROOT"

nvidia-smi

test -f "$DATA"

MAIN_STATE=$(sacct -j "$MAIN_JOB" --format=State --noheader | head -1 | awk '{print $1}')
echo "MAIN_STATE=$MAIN_STATE"

case "$MAIN_STATE" in
  COMPLETED|TIMEOUT)
    echo "Main job state allows continuation check."
    ;;
  *)
    echo "Main job state is $MAIN_STATE. Refusing to continue."
    exit 1
    ;;
esac

LATEST=$(find "$OUTROOT" -name "network-snapshot-*.pkl" | sort | tail -1)

if [ -z "$LATEST" ]; then
  echo "No snapshot found under $OUTROOT. Cannot continue."
  exit 1
fi

echo "LATEST=$LATEST"

BASE=$(basename "$LATEST")
CUR_KIMG=$(echo "$BASE" | sed -E 's/network-snapshot-0*([0-9]+)\.pkl/\1/')

if [ -z "$CUR_KIMG" ]; then
  echo "Failed to parse kimg from $BASE"
  exit 1
fi

echo "CUR_KIMG=$CUR_KIMG"
echo "TARGET_KIMG=$TARGET_KIMG"

if [ "$CUR_KIMG" -ge "$TARGET_KIMG" ]; then
  echo "Already reached target kimg. Nothing to do."
  exit 0
fi

REMAIN=$((TARGET_KIMG - CUR_KIMG))
echo "REMAIN_KIMG=$REMAIN"

python train.py \
  --outdir="$CONT_OUTROOT" \
  --data="$DATA" \
  --resume="$LATEST" \
  --cfg=paper256 \
  --gpus=8 \
  --batch=128 \
  --gamma=2 \
  --mirror=1 \
  --aug=ada \
  --augpipe=color \
  --target=0.6 \
  --snap=25 \
  --kimg="$REMAIN" \
  --metrics=none \
  --workers=8

echo "Done RAWFRAME 1F P50 continuation: $(date)"
