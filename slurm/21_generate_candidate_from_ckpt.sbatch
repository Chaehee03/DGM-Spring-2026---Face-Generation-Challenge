#!/bin/bash
#SBATCH --job-name=dgm_gen_ckpt
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

: "${CKPT:?CKPT is required}"
: "${TAG:?TAG is required}"
: "${MODE:?MODE is required}"   # pure or mixed

OUTROOT=/data/chsong/dgm-face/submissions/$TAG
IMGDIR=$OUTROOT/images

mkdir -p "$IMGDIR"

echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CKPT=$CKPT"
echo "TAG=$TAG"
echo "MODE=$MODE"
echo "OUTROOT=$OUTROOT"

nvidia-smi

if [ "$MODE" = "pure" ]; then
    : "${PSI:?PSI is required for pure mode}"
    echo "Pure generation with psi=$PSI"
    python generate.py \
      --network="$CKPT" \
      --seeds=0-999 \
      --trunc="$PSI" \
      --noise-mode=const \
      --outdir="$IMGDIR"

    echo "$CKPT" > "$OUTROOT/checkpoint.txt"
    echo "$PSI"  > "$OUTROOT/truncation.txt"
    echo "0-999" > "$OUTROOT/seeds.txt"

elif [ "$MODE" = "mixed" ]; then
    : "${PSI_A:?PSI_A is required}"
    : "${PSI_B:?PSI_B is required}"
    : "${N_A:?N_A is required}"   # first count, second is 1000-N_A

    N_B=$((1000 - N_A))
    START_B=$N_A
    END_A=$((N_A - 1))
    END_B=999

    echo "Mixed generation:"
    echo "  first  ${N_A} images -> psi=${PSI_A}, seeds=0-${END_A}"
    echo "  second ${N_B} images -> psi=${PSI_B}, seeds=${START_B}-${END_B}"

    python generate.py \
      --network="$CKPT" \
      --seeds=0-${END_A} \
      --trunc="$PSI_A" \
      --noise-mode=const \
      --outdir="$IMGDIR"

    python generate.py \
      --network="$CKPT" \
      --seeds=${START_B}-${END_B} \
      --trunc="$PSI_B" \
      --noise-mode=const \
      --outdir="$IMGDIR"

    echo "$CKPT" > "$OUTROOT/checkpoint.txt"
    printf "first %d -> psi=%s\nsecond %d -> psi=%s\n" "$N_A" "$PSI_A" "$N_B" "$PSI_B" > "$OUTROOT/truncation.txt"
    printf "0-%d and %d-999\n" "$END_A" "$START_B" > "$OUTROOT/seeds.txt"

else
    echo "Unknown MODE=$MODE"
    exit 1
fi

echo "Image count:"
find "$IMGDIR" -maxdepth 1 -type f -name "*.png" | wc -l

cd /workspace/chsong/dgm-face
python scripts/make_submission_zip.py \
  --input-dir "$IMGDIR" \
  --output-zip "$OUTROOT/submission.zip"

echo "Done: $(date)"
