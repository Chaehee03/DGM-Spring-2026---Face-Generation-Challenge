#!/bin/bash
#SBATCH --job-name=dgm_raw3f_gen
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%A_%a.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

RUN=$(ls -dt \
  /data/chsong/dgm-face/training-runs-rawframe-ffhq-ada-color-k2000/* \
  /data/chsong/dgm-face/training-runs-rawframe-ffhq-ada-color-k2000-cont/* \
  2>/dev/null | head -n 1)

if [ -z "${RUN:-}" ]; then
  echo "Could not find rawframe 3f run directory."
  exit 1
fi

DATE_TAG=20260601
ROOT=/data/chsong/dgm-face/submissions/${DATE_TAG}_raw3f_top4
mkdir -p "$ROOT"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    NAME="k1945_pure075"
    CKPT="$RUN/network-snapshot-001945.pkl"
    MODE="pure075"
    ;;
  1)
    NAME="k1945_mix700_075_300_090"
    CKPT="$RUN/network-snapshot-001945.pkl"
    MODE="mix700"
    ;;
  2)
    NAME="k1843_pure075"
    CKPT="$RUN/network-snapshot-001843.pkl"
    MODE="pure075"
    ;;
  3)
    NAME="k2000_pure075"
    CKPT="$RUN/network-snapshot-002000.pkl"
    MODE="pure075"
    ;;
  *)
    echo "Invalid task id"
    exit 1
    ;;
esac

test -f "$CKPT"

CAND="$ROOT/$NAME"
IMGDIR="$CAND/images"
mkdir -p "$IMGDIR"

rm -f "$IMGDIR"/*.png
rm -f "$CAND/submission.zip" "$CAND/meta.txt"

echo "Start: $(date)"
echo "RUN=$RUN"
echo "NAME=$NAME"
echo "CKPT=$CKPT"
echo "MODE=$MODE"
echo "IMGDIR=$IMGDIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi

if [ "$MODE" = "pure075" ]; then
  python generate.py \
    --network="$CKPT" \
    --seeds=0-999 \
    --trunc=0.75 \
    --noise-mode=const \
    --outdir="$IMGDIR"

elif [ "$MODE" = "mix700" ]; then
  python generate.py \
    --network="$CKPT" \
    --seeds=0-699 \
    --trunc=0.75 \
    --noise-mode=const \
    --outdir="$IMGDIR"

  python generate.py \
    --network="$CKPT" \
    --seeds=700-999 \
    --trunc=0.90 \
    --noise-mode=const \
    --outdir="$IMGDIR"
else
  echo "Unknown MODE=$MODE"
  exit 1
fi

python - <<PY
from pathlib import Path
import zipfile

cand = Path("$CAND")
imgdir = cand / "images"
imgs = sorted(imgdir.glob("*.png"))

print("generated images:", len(imgs))
if len(imgs) != 1000:
    raise SystemExit(f"Expected 1000 images, got {len(imgs)}")

meta = cand / "meta.txt"
meta.write_text(
    "candidate=$NAME\n"
    "checkpoint=$CKPT\n"
    "mode=$MODE\n"
    "seed_policy=pure:0-999 or mix:0-699@0.75,700-999@0.90\n"
)

zpath = cand / "submission.zip"
if zpath.exists():
    zpath.unlink()

with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
    for i, p in enumerate(imgs):
        z.write(p, arcname=f"img_{i:04d}.png")

with zipfile.ZipFile(zpath, "r") as z:
    names = z.namelist()
    img_names = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    nested = [n for n in img_names if "/" in n.strip("/")]

print("zip image count:", len(img_names))
print("zip nested count:", len(nested))
print("zip size MB:", round(zpath.stat().st_size / 1024 / 1024, 2))

if len(img_names) != 1000:
    raise SystemExit(f"Zip does not contain 1000 images: {len(img_names)}")
if len(nested) != 0:
    raise SystemExit(f"Zip contains nested paths: {nested[:5]}")

print("Done packaging:", zpath)
PY

echo "Done: $(date)"
