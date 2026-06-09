#!/bin/bash
#SBATCH --job-name=dgm_1f_p095
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

CKPT=/data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000/00000-celebvhq_rawframe_1f_p50_center256_jpg-mirror-paper256-gamma2-kimg2000-batch128-ada-target0.6-color-resumecustom/network-snapshot-001843.pkl

ROOT=/data/chsong/dgm-face/submissions/20260602_1f_k1843_pure095
FINAL=$ROOT/1f_k1843_pure095
IMGDIR=$FINAL/images

rm -rf "$ROOT"
mkdir -p "$IMGDIR"

echo "Start: $(date)"
echo "Candidate=1f_k1843_pure095"
echo "CKPT=$CKPT"
echo "ROOT=$ROOT"

test -f "$CKPT"

nvidia-smi

# 8 GPU 병렬 생성, 총 1000장
CUDA_VISIBLE_DEVICES=0 python generate.py --network="$CKPT" --outdir="$IMGDIR" --trunc=0.95 --noise-mode=const --seeds=0-124 &
CUDA_VISIBLE_DEVICES=1 python generate.py --network="$CKPT" --outdir="$IMGDIR" --trunc=0.95 --noise-mode=const --seeds=125-249 &
CUDA_VISIBLE_DEVICES=2 python generate.py --network="$CKPT" --outdir="$IMGDIR" --trunc=0.95 --noise-mode=const --seeds=250-374 &
CUDA_VISIBLE_DEVICES=3 python generate.py --network="$CKPT" --outdir="$IMGDIR" --trunc=0.95 --noise-mode=const --seeds=375-499 &
CUDA_VISIBLE_DEVICES=4 python generate.py --network="$CKPT" --outdir="$IMGDIR" --trunc=0.95 --noise-mode=const --seeds=500-624 &
CUDA_VISIBLE_DEVICES=5 python generate.py --network="$CKPT" --outdir="$IMGDIR" --trunc=0.95 --noise-mode=const --seeds=625-749 &
CUDA_VISIBLE_DEVICES=6 python generate.py --network="$CKPT" --outdir="$IMGDIR" --trunc=0.95 --noise-mode=const --seeds=750-874 &
CUDA_VISIBLE_DEVICES=7 python generate.py --network="$CKPT" --outdir="$IMGDIR" --trunc=0.95 --noise-mode=const --seeds=875-999 &

wait

echo "Generation done: $(date)"

python - <<PY
from pathlib import Path
import zipfile
import re

name = "1f_k1843_pure095"
final = Path("$FINAL")
imgdir = Path("$IMGDIR")
zpath = final / f"{name}.zip"

def seed_key(p):
    m = re.search(r"(\\d+)", p.stem)
    return int(m.group(1)) if m else 10**18

imgs = sorted(imgdir.glob("*.png"), key=seed_key)

print("image_count:", len(imgs))
if len(imgs) != 1000:
    raise SystemExit(f"expected 1000 images, got {len(imgs)}")

meta = final / "meta.txt"
meta.write_text(
    f"candidate={name}\\n"
    f"frame=1f_p50\\n"
    f"effective_kimg=1843\\n"
    f"checkpoint=$CKPT\\n"
    f"psi=0.95 pure\\n"
    f"noise_mode=const\\n"
    f"seeds=0-999\\n"
)

if zpath.exists():
    zpath.unlink()

# zip root에 이미지만 들어가게 저장
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
