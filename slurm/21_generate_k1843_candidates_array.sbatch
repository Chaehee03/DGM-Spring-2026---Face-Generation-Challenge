#!/bin/bash
#SBATCH --job-name=dgm_k1843_gen
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%A_%a.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%A_%a.err

set -e
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

RUN=/data/chsong/dgm-face/training-runs-ffhq-ada-color-clean-k2000/00000-celebvhq_256_insight_3f_std-mirror-paper256-gamma2-kimg2000-batch128-ada-target0.6-color-resumecustom
CKPT=$RUN/network-snapshot-001843.pkl
DATE_TAG=20260601
ROOT=/data/chsong/dgm-face/submissions/${DATE_TAG}_k1843_candidates

test -f "$CKPT"
mkdir -p "$ROOT"

echo "Start: $(date)"
echo "Job: $SLURM_JOB_ID / Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CKPT=$CKPT"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    NAME=k1843_pure075
    OUT=$ROOT/$NAME
    mkdir -p "$OUT/images"
    python generate.py --network="$CKPT" --outdir="$OUT/images" --seeds=0-999 --trunc=0.75
    DESC="1000 images, psi=0.75, seeds=0-999"
    ;;
  1)
    NAME=k1843_pure085
    OUT=$ROOT/$NAME
    mkdir -p "$OUT/images"
    python generate.py --network="$CKPT" --outdir="$OUT/images" --seeds=0-999 --trunc=0.85
    DESC="1000 images, psi=0.85, seeds=0-999"
    ;;
  2)
    NAME=k1843_mix700_075_300_090
    OUT=$ROOT/$NAME
    mkdir -p "$OUT/images"
    python generate.py --network="$CKPT" --outdir="$OUT/images" --seeds=0-699 --trunc=0.75
    python generate.py --network="$CKPT" --outdir="$OUT/images" --seeds=700-999 --trunc=0.90
    DESC="700 images psi=0.75 seeds=0-699 + 300 images psi=0.90 seeds=700-999"
    ;;
  3)
    NAME=k1843_pure065
    OUT=$ROOT/$NAME
    mkdir -p "$OUT/images"
    python generate.py --network="$CKPT" --outdir="$OUT/images" --seeds=0-999 --trunc=0.65
    DESC="1000 images, psi=0.65, seeds=0-999"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

python - <<PY
from pathlib import Path
img_dir = Path("$OUT/images")
imgs = sorted(img_dir.glob("*.png"))
print("generated images:", len(imgs))
if len(imgs) != 1000:
    raise SystemExit(f"Expected 1000 images, got {len(imgs)}")
PY

cat > "$OUT/manifest.txt" <<MANIFEST
candidate=$NAME
date_tag=$DATE_TAG
checkpoint=$CKPT
description=$DESC
MANIFEST

cd "$OUT/images"
zip -q -j "$OUT/${NAME}_${DATE_TAG}.zip" *.png

python - <<PY
import zipfile
from pathlib import Path
p = Path("$OUT/${NAME}_${DATE_TAG}.zip")
with zipfile.ZipFile(p, "r") as z:
    imgs = [x for x in z.namelist() if x.lower().endswith((".png", ".jpg", ".jpeg"))]
    nested = [x for x in imgs if "/" in x.strip("/")]
print("zip:", p)
print("zip image count:", len(imgs))
print("nested image count:", len(nested))
print("zip size MB:", round(p.stat().st_size / 1024 / 1024, 2))
if len(imgs) != 1000:
    raise SystemExit(f"Zip does not contain 1000 images: {len(imgs)}")
if len(nested) != 0:
    raise SystemExit(f"Zip has nested images: {len(nested)}")
PY

echo "Done: $(date)"
