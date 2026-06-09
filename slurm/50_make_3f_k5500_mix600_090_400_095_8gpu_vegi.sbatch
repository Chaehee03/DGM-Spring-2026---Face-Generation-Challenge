#!/bin/bash
#SBATCH --job-name=dgm_3f5500_m6095
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

export TORCH_EXTENSIONS_DIR=/data/chsong/torch_extensions_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"

CKPT=/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k5500-from2000/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg3500-batch128-ada-target0.6-color-resumecustom/network-snapshot-003500.pkl

ROOT=/data/chsong/dgm-face/submissions/20260603_3f_k5500_mix600_090_400_095
CAND=3f_k5500_mix600_090_400_095
OUTDIR=$ROOT/$CAND
IMGDIR=$OUTDIR/images

rm -rf "$OUTDIR"
mkdir -p "$IMGDIR"

echo "Start: $(date)"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-none}"
echo "CKPT=$CKPT"
echo "OUTDIR=$OUTDIR"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

nvidia-smi || true

echo "[1/8] psi0.90 seeds 0-149 on GPU0"
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --network="$CKPT" \
  --seeds=0-149 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$IMGDIR" &
P1=$!

echo "[2/8] psi0.90 seeds 150-299 on GPU1"
CUDA_VISIBLE_DEVICES=1 python generate.py \
  --network="$CKPT" \
  --seeds=150-299 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$IMGDIR" &
P2=$!

echo "[3/8] psi0.90 seeds 300-449 on GPU2"
CUDA_VISIBLE_DEVICES=2 python generate.py \
  --network="$CKPT" \
  --seeds=300-449 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$IMGDIR" &
P3=$!

echo "[4/8] psi0.90 seeds 450-599 on GPU3"
CUDA_VISIBLE_DEVICES=3 python generate.py \
  --network="$CKPT" \
  --seeds=450-599 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$IMGDIR" &
P4=$!

echo "[5/8] psi0.95 seeds 600-699 on GPU4"
CUDA_VISIBLE_DEVICES=4 python generate.py \
  --network="$CKPT" \
  --seeds=600-699 \
  --trunc=0.95 \
  --noise-mode=const \
  --outdir="$IMGDIR" &
P5=$!

echo "[6/8] psi0.95 seeds 700-799 on GPU5"
CUDA_VISIBLE_DEVICES=5 python generate.py \
  --network="$CKPT" \
  --seeds=700-799 \
  --trunc=0.95 \
  --noise-mode=const \
  --outdir="$IMGDIR" &
P6=$!

echo "[7/8] psi0.95 seeds 800-899 on GPU6"
CUDA_VISIBLE_DEVICES=6 python generate.py \
  --network="$CKPT" \
  --seeds=800-899 \
  --trunc=0.95 \
  --noise-mode=const \
  --outdir="$IMGDIR" &
P7=$!

echo "[8/8] psi0.95 seeds 900-999 on GPU7"
CUDA_VISIBLE_DEVICES=7 python generate.py \
  --network="$CKPT" \
  --seeds=900-999 \
  --trunc=0.95 \
  --noise-mode=const \
  --outdir="$IMGDIR" &
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
import zipfile
import re

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

meta = outdir / "meta.txt"
meta.write_text(
    f"candidate={cand}\\n"
    f"source=3f_rawframe_0205080\\n"
    f"checkpoint=$CKPT\\n"
    f"effective_kimg=5500\\n"
    f"checkpoint_file_kimg=3500_cont_from_2000\\n"
    f"mix=600@psi0.90 + 400@psi0.95\\n"
    f"seeds=0-599@0.90,600-999@0.95\\n"
    f"noise_mode=const\\n"
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

print("ZIP_VALIDATION_OK")
PY

echo "Done: $(date)"
find "$ROOT" -maxdepth 3 -type f \( -name "*.zip" -o -name "meta.txt" \) -print | sort
