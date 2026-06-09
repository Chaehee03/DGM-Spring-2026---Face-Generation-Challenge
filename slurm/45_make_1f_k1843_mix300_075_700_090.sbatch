#!/bin/bash
#SBATCH --job-name=dgm_1f_m3070
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
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

CKPT_1F=$(find \
  /data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000 \
  /data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000-cont \
  -name "network-snapshot-001843.pkl" 2>/dev/null | sort | tail -n 1)

echo "CKPT_1F=$CKPT_1F"
test -f "$CKPT_1F"

if [[ "$CKPT_1F" != *"network-snapshot-001843.pkl" ]]; then
  echo "ERROR: wrong checkpoint: $CKPT_1F"
  exit 1
fi

ROOT=/data/chsong/dgm-face/submissions/20260602_1f_k1843_mix300_075_700_090
CAND=$ROOT/1f_k1843_mix300_075_700_090

rm -rf "$ROOT"
mkdir -p "$CAND/images"

echo "Start generation: $(date)"
echo "Candidate: 1f_k1843_mix300_075_700_090"
echo "Mix: 300@psi0.75 + 700@psi0.90"
echo "Output: $CAND"

nvidia-smi

# 300장: psi 0.75
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --network="$CKPT_1F" \
  --seeds=0-299 \
  --trunc=0.75 \
  --noise-mode=const \
  --outdir="$CAND/images" &

# 700장: psi 0.90
CUDA_VISIBLE_DEVICES=1 python generate.py \
  --network="$CKPT_1F" \
  --seeds=300-999 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$CAND/images" &

wait

echo "Generation done: $(date)"

python - <<PY
from pathlib import Path
import zipfile
import re

name = "1f_k1843_mix300_075_700_090"
outdir = Path("$CAND")
imgdir = outdir / "images"
ckpt = "$CKPT_1F"

def seed_key(p):
    m = re.search(r"(\\d+)", p.stem)
    return int(m.group(1)) if m else 10**18

imgs = sorted(imgdir.glob("*.png"), key=seed_key)

print("image_count:", len(imgs))
if len(imgs) != 1000:
    raise SystemExit(f"expected 1000 images, got {len(imgs)}")

meta = outdir / "meta.txt"
meta.write_text(
    f"candidate={name}\\n"
    f"frame=1f_p50\\n"
    f"effective_kimg=1843\\n"
    f"checkpoint={ckpt}\\n"
    f"mix=300@psi0.75 + 700@psi0.90\\n"
    f"noise_mode=const\\n"
    f"seeds=0-999\\n"
)

zpath = outdir / f"{name}.zip"
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

echo "All done: $(date)"
echo "Output:"
find "$ROOT" -maxdepth 2 -type f \( -name "*.zip" -o -name "meta.txt" \) -print | sort
