#!/bin/bash
#SBATCH --job-name=dgm_1f1843_m8095
#SBATCH --partition=R4090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

CKPT=/data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000/00000-celebvhq_rawframe_1f_p50_center256_jpg-mirror-paper256-gamma2-kimg2000-batch128-ada-target0.6-color-resumecustom/network-snapshot-001843.pkl

ROOT=/data/chsong/dgm-face/submissions/20260602_1fp50_k1843_mix800_090_200_095
CAND=$ROOT/1f_k1843_mix800_090_200_095
IMGDIR=$CAND/images

rm -rf "$CAND"
mkdir -p "$IMGDIR"

echo "Start: $(date)"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CKPT=$CKPT"
echo "CAND=$CAND"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

nvidia-smi

# 800 images: psi=0.90, seeds 0-799
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --network="$CKPT" \
  --outdir="$IMGDIR" \
  --seeds=0-799 \
  --trunc=0.90 \
  --noise-mode=const &

# 200 images: psi=0.95, seeds 800-999
CUDA_VISIBLE_DEVICES=1 python generate.py \
  --network="$CKPT" \
  --outdir="$IMGDIR" \
  --seeds=800-999 \
  --trunc=0.95 \
  --noise-mode=const &

wait

python - <<PY
from pathlib import Path
import zipfile
import re

name = "1f_k1843_mix800_090_200_095"
cand = Path("$CAND")
imgdir = Path("$IMGDIR")
zpath = cand / f"{name}.zip"

def seed_key(p):
    m = re.search(r"(\\d+)", p.stem)
    return int(m.group(1)) if m else 10**18

imgs = sorted(imgdir.glob("*.png"), key=seed_key)
print("image_count:", len(imgs))

if len(imgs) != 1000:
    raise SystemExit(f"expected 1000 images, got {len(imgs)}")

meta = cand / "meta.txt"
meta.write_text(
    f"candidate={name}\\n"
    f"frame=1f_p50\\n"
    f"effective_kimg=1843\\n"
    f"checkpoint=$CKPT\\n"
    f"mix=800@psi0.90 + 200@psi0.95\\n"
    f"noise_mode=const\\n"
    f"seeds=0-799@0.90,800-999@0.95\\n"
)

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
print("images:", len(image_names))
print("nested:", len(nested))
print("non_imgs:", len(non_imgs))
print("size MB:", round(size_mb, 2))

if len(image_names) != 1000:
    raise SystemExit("zip image count is not 1000")
if nested:
    raise SystemExit("zip has nested paths")
if non_imgs:
    raise SystemExit("zip has non-image files")
if size_mb >= 200:
    raise SystemExit("zip exceeds 200MB")

print("ZIP_OK")
PY

echo "Done: $(date)"
find "$ROOT" -maxdepth 2 -type f \( -name "*.zip" -o -name "meta.txt" \) -print | sort
