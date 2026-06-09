#!/bin/bash
#SBATCH --job-name=dgm_1f1843_fix
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

CKPT=$(find /data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000 \
  /data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000-cont \
  -name "network-snapshot-001843.pkl" 2>/dev/null | sort | tail -n 1)

if [ -z "$CKPT" ]; then
  echo "ERROR: k1843 checkpoint not found"
  exit 1
fi

test -f "$CKPT"

ROOT=/data/chsong/dgm-face/submissions/20260601_1fp50_k1843_FIXED
PURE=$ROOT/k1843_pure075
MIX=$ROOT/k1843_mixed700_075_300_090

rm -rf "$ROOT"
mkdir -p "$PURE/images" "$MIX/images"

echo "Start: $(date)"
echo "CKPT=$CKPT"
echo "ROOT=$ROOT"
echo "This run must use network-snapshot-001843.pkl only."

if [[ "$CKPT" != *"network-snapshot-001843.pkl" ]]; then
  echo "ERROR: wrong checkpoint: $CKPT"
  exit 1
fi

nvidia-smi

echo "=== generate k1843 pure075 ==="
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --network="$CKPT" \
  --seeds=0-499 \
  --trunc=0.75 \
  --noise-mode=const \
  --outdir="$PURE/images" &

CUDA_VISIBLE_DEVICES=1 python generate.py \
  --network="$CKPT" \
  --seeds=500-999 \
  --trunc=0.75 \
  --noise-mode=const \
  --outdir="$PURE/images" &

echo "=== generate k1843 mixed700_075_300_090 ==="
CUDA_VISIBLE_DEVICES=2 python generate.py \
  --network="$CKPT" \
  --seeds=0-699 \
  --trunc=0.75 \
  --noise-mode=const \
  --outdir="$MIX/images" &

CUDA_VISIBLE_DEVICES=3 python generate.py \
  --network="$CKPT" \
  --seeds=700-999 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$MIX/images" &

wait

python - <<PY
from pathlib import Path
import zipfile, re

targets = [
    ("k1843_pure075", Path("$PURE")),
    ("k1843_mixed700_075_300_090", Path("$MIX")),
]

def seed_key(p):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 10**18

for name, root in targets:
    imgdir = root / "images"
    imgs = sorted(imgdir.glob("*.png"), key=seed_key)

    print(name, "image_count:", len(imgs))
    if len(imgs) != 1000:
        raise SystemExit(f"{name}: expected 1000 images, got {len(imgs)}")

    meta = root / "meta.txt"
    meta.write_text(
        f"candidate={name}\n"
        f"checkpoint=$CKPT\n"
        f"kimg=1843\n"
        f"pure_or_mixed={name}\n"
    )

    zpath = root / f"{name}.zip"
    if zpath.exists():
        zpath.unlink()

    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for i, p in enumerate(imgs):
            z.write(p, arcname=f"img_{i:04d}.png")

    with zipfile.ZipFile(zpath, "r") as z:
        names = z.namelist()
        image_names = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
        nested = [n for n in image_names if "/" in n.strip("/")]
        non_imgs = [n for n in names if not n.lower().endswith((".png", ".jpg", ".jpeg"))]

    size_mb = zpath.stat().st_size / 1024 / 1024

    print("zip:", zpath)
    print("  images:", len(image_names))
    print("  nested:", len(nested))
    print("  non_imgs:", len(non_imgs))
    print("  size MB:", round(size_mb, 2))

    if len(image_names) != 1000:
        raise SystemExit(f"{name}: zip does not contain 1000 images")
    if nested:
        raise SystemExit(f"{name}: zip contains nested paths")
    if non_imgs:
        raise SystemExit(f"{name}: zip contains non-image files")
    if size_mb >= 200:
        raise SystemExit(f"{name}: zip exceeds 200MB")

print("ALL_ZIP_OK")
PY

echo "Done: $(date)"
