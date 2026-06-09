#!/bin/bash
#SBATCH --job-name=dgm_1f1843_sub2
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

if [ -z "$CKPT" ]; then
  echo "CKPT is empty"
  exit 1
fi

test -f "$CKPT"

DATE_TAG=$(date +%Y%m%d)
ROOT=/data/chsong/dgm-face/submissions/${DATE_TAG}_1fp50_k1843_submit2

PURE=$ROOT/k1843_pure075
MIX=$ROOT/k1843_mixed700_075_300_090

mkdir -p "$PURE/images" "$MIX/images"

echo "Start: $(date)"
echo "CKPT=$CKPT"
echo "ROOT=$ROOT"

echo "=== generate pure075 ==="
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --network="$CKPT" \
  --seeds=0-499 \
  --trunc=0.75 \
  --outdir="$PURE/images" &

CUDA_VISIBLE_DEVICES=1 python generate.py \
  --network="$CKPT" \
  --seeds=500-999 \
  --trunc=0.75 \
  --outdir="$PURE/images" &

echo "=== generate mixed ==="
CUDA_VISIBLE_DEVICES=2 python generate.py \
  --network="$CKPT" \
  --seeds=0-699 \
  --trunc=0.75 \
  --outdir="$MIX/images" &

CUDA_VISIBLE_DEVICES=3 python generate.py \
  --network="$CKPT" \
  --seeds=700-999 \
  --trunc=0.90 \
  --outdir="$MIX/images" &

wait

echo "=== validate image counts ==="
python - <<PY
from pathlib import Path
for d in [
    Path("$PURE/images"),
    Path("$MIX/images"),
]:
    imgs = sorted(d.glob("*.png"))
    print(d, "->", len(imgs))
    if len(imgs) != 1000:
        raise SystemExit(f"Expected 1000 images in {d}, got {len(imgs)}")
PY

echo "=== make submission zips (flat root, no subfolders) ==="
python - <<PY
from pathlib import Path
import zipfile
import re

def seed_key(p):
    m = re.search(r'(\d+)', p.stem)
    return int(m.group(1)) if m else 10**18

targets = [
    Path("$PURE"),
    Path("$MIX"),
]

for root in targets:
    img_dir = root / "images"
    imgs = sorted(img_dir.glob("*.png"), key=seed_key)

    if len(imgs) != 1000:
        raise SystemExit(f"Expected 1000 images, got {len(imgs)} for {root}")

    zpath = root / "submission.zip"
    if zpath.exists():
        zpath.unlink()

    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for p in imgs:
            z.write(p, arcname=p.name)  # zip 루트에 바로 저장

    with zipfile.ZipFile(zpath, "r") as z:
        names = z.namelist()
        imgs_in_zip = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
        nested = [n for n in imgs_in_zip if "/" in n.strip("/")]

    print("zip:", zpath)
    print("  image count:", len(imgs_in_zip))
    print("  nested files:", len(nested))
    print("  size MB:", round(zpath.stat().st_size / 1024 / 1024, 2))

    if len(imgs_in_zip) != 1000:
        raise SystemExit(f"Zip does not contain 1000 images: {zpath}")
    if len(nested) != 0:
        raise SystemExit(f"Zip contains nested paths: {zpath}")
PY

echo "=== final outputs ==="
ls -lh "$PURE/submission.zip" "$MIX/submission.zip"

echo "Done: $(date)"
