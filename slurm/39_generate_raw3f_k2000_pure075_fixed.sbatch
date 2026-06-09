#!/bin/bash
#SBATCH --job-name=dgm_raw3f_k2000
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-03:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

CKPT=/data/chsong/dgm-face/training-runs-rawframe-ffhq-ada-color-k2000/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg2000-batch128-ada-target0.6-color-resumecustom/network-snapshot-002000.pkl

ROOT=/data/chsong/dgm-face/submissions/20260601_raw3f_k2000_pure075_fixed
CAND="$ROOT/raw3f_k2000_pure075"
IMGDIR="$CAND/images"
ZIPPATH="$CAND/raw3f_k2000_pure075.zip"
META="$CAND/meta.txt"

rm -rf "$CAND"
mkdir -p "$IMGDIR"

echo "Start generation: $(date)"
echo "CKPT=$CKPT"
echo "IMGDIR=$IMGDIR"
echo "ZIPPATH=$ZIPPATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

test -f "$CKPT"
nvidia-smi

python generate.py \
  --network="$CKPT" \
  --seeds=0-999 \
  --trunc=0.75 \
  --noise-mode=const \
  --outdir="$IMGDIR"

python - <<PY
from pathlib import Path
import zipfile

imgdir = Path("$IMGDIR")
zpath = Path("$ZIPPATH")
meta = Path("$META")

imgs = sorted(imgdir.glob("*.png"))
print("generated images:", len(imgs))

if len(imgs) != 1000:
    raise SystemExit(f"Expected 1000 images, got {len(imgs)}")

meta.write_text(
    "candidate=raw3f_k2000_pure075\n"
    "checkpoint=$CKPT\n"
    "kimg=2000\n"
    "psi=0.75\n"
    "seeds=0-999\n"
    "noise_mode=const\n"
)

if zpath.exists():
    zpath.unlink()

with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
    for i, p in enumerate(imgs):
        z.write(p, arcname=f"img_{i:04d}.png")

with zipfile.ZipFile(zpath, "r") as z:
    names = z.namelist()
    img_names = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    nested = [n for n in img_names if "/" in n.strip("/")]
    non_imgs = [n for n in names if not n.lower().endswith((".png", ".jpg", ".jpeg"))]

print("zip image count:", len(img_names))
print("zip nested count:", len(nested))
print("zip non-image count:", len(non_imgs))
print("zip size MB:", round(zpath.stat().st_size / 1024 / 1024, 2))

if len(img_names) != 1000:
    raise SystemExit(f"Zip does not contain 1000 images: {len(img_names)}")
if nested:
    raise SystemExit(f"Zip contains nested paths: {nested[:5]}")
if non_imgs:
    raise SystemExit(f"Zip contains non-image files: {non_imgs[:5]}")
if zpath.stat().st_size >= 200 * 1024 * 1024:
    raise SystemExit("Zip exceeds 200MB")

print("ZIP_OK", zpath)
PY

echo "Done generation: $(date)"
