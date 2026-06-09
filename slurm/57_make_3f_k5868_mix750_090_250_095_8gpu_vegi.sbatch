#!/bin/bash
#SBATCH --job-name=dgm3f5868_m7595
#SBATCH --partition=R4090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

export TORCH_CUDA_ARCH_LIST="8.9"
export TORCH_EXTENSIONS_DIR=/data/chsong/torch_extensions_stylegan2ada_py39_cu118_sm89
export MAX_JOBS=8

# 5500 + 368 = 5868
CKPT=/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k6500-from5500/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg1000-batch128-ada-target0.6-color-resumecustom/network-snapshot-000368.pkl

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

ROOT=/data/chsong/dgm-face/submissions/20260608_3f_k5868_mix750_090_250_095
CAND=3f_k5868_mix750_090_250_095
OUTDIR=$ROOT/$CAND
IMGDIR=$OUTDIR/images

rm -rf "$OUTDIR"
mkdir -p "$IMGDIR"

echo "Start: $(date)"
echo "HOST=$(hostname)"
echo "CKPT=$CKPT"
echo "OUTDIR=$OUTDIR"

# 750장 @ psi=0.90
CUDA_VISIBLE_DEVICES=0 python generate.py --network="$CKPT" --seeds=0-187   --trunc=0.90 --noise-mode=const --outdir="$IMGDIR" &
P1=$!
CUDA_VISIBLE_DEVICES=1 python generate.py --network="$CKPT" --seeds=188-375 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR" &
P2=$!
CUDA_VISIBLE_DEVICES=2 python generate.py --network="$CKPT" --seeds=376-562 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR" &
P3=$!
CUDA_VISIBLE_DEVICES=3 python generate.py --network="$CKPT" --seeds=563-749 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR" &
P4=$!

# 250장 @ psi=0.95
CUDA_VISIBLE_DEVICES=4 python generate.py --network="$CKPT" --seeds=750-812 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR" &
P5=$!
CUDA_VISIBLE_DEVICES=5 python generate.py --network="$CKPT" --seeds=813-874 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR" &
P6=$!
CUDA_VISIBLE_DEVICES=6 python generate.py --network="$CKPT" --seeds=875-937 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR" &
P7=$!
CUDA_VISIBLE_DEVICES=7 python generate.py --network="$CKPT" --seeds=938-999 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR" &
P8=$!

wait $P1
wait $P2
wait $P3
wait $P4
wait $P5
wait $P6
wait $P7
wait $P8

echo "Generation finished: $(date)"

python3 - <<PY
from pathlib import Path
import zipfile, re

cand = "$CAND"
outdir = Path("$OUTDIR")
imgdir = Path("$IMGDIR")
zpath = outdir / f"{cand}.zip"

def seed_key(p):
    m = re.search(r"(\\d+)", p.stem)
    return int(m.group(1)) if m else 10**18

imgs = sorted(
    [p for p in imgdir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]],
    key=seed_key,
)

print("generated images:", len(imgs))
if len(imgs) != 1000:
    raise SystemExit(f"Expected 1000 images, got {len(imgs)}")

if zpath.exists():
    zpath.unlink()

with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
    for i, p in enumerate(imgs):
        z.write(p, arcname=f"img_{i:04d}.png")

with zipfile.ZipFile(zpath, "r") as z:
    names = z.namelist()
    imgs_in_zip = [n for n in names if n.lower().endswith((".png",".jpg",".jpeg"))]
    nested = [n for n in imgs_in_zip if "/" in n.strip("/")]
    non_imgs = [n for n in names if not n.lower().endswith((".png",".jpg",".jpeg"))]

size_mb = zpath.stat().st_size / 1024 / 1024

print("zip:", zpath)
print("images:", len(imgs_in_zip))
print("nested:", len(nested))
print("non_imgs:", len(non_imgs))
print("size MB:", round(size_mb, 2))

if len(imgs_in_zip) != 1000:
    raise SystemExit("zip image count is not 1000")
if nested:
    raise SystemExit("zip has nested paths")
if non_imgs:
    raise SystemExit("zip has non-image files")
if size_mb >= 200:
    raise SystemExit("zip exceeds 200MB")

print("ZIP_VALIDATION_OK")
PY

echo "Done: $(date)"
