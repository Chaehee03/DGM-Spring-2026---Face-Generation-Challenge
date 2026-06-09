#!/bin/bash
#SBATCH --job-name=dgm_1f1843_m7095
#SBATCH --partition=R4090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

CKPT=/data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000/00000-celebvhq_rawframe_1f_p50_center256_jpg-mirror-paper256-gamma2-kimg2000-batch128-ada-target0.6-color-resumecustom/network-snapshot-001843.pkl

ROOT=/data/chsong/dgm-face/submissions/20260603_1fp50_k1843_mix700_090_300_095
CAND=1f_k1843_mix700_090_300_095
OUTDIR=$ROOT/$CAND
IMGDIR=$OUTDIR/images

mkdir -p "$IMGDIR"

echo "Start: $(date)"
echo "Checkpoint: $CKPT"
echo "Output dir: $OUTDIR"
nvidia-smi || true

test -f "$CKPT"

# -------------------------
# 700 images: psi=0.90, seeds 0-699
# 300 images: psi=0.95, seeds 700-999
# 두 GPU에 나눠서 동시에 실행
# -------------------------

CUDA_VISIBLE_DEVICES=0 python generate.py \
  --network="$CKPT" \
  --seeds=0-699 \
  --trunc=0.90 \
  --outdir="$IMGDIR" &

PID1=$!

CUDA_VISIBLE_DEVICES=1 python generate.py \
  --network="$CKPT" \
  --seeds=700-999 \
  --trunc=0.95 \
  --outdir="$IMGDIR" &

PID2=$!

wait $PID1
wait $PID2

echo "Generation finished: $(date)"

# -------------------------
# 이미지 개수 검증
# -------------------------
python3 - <<PY
from pathlib import Path
imgdir = Path("$IMGDIR")
imgs = sorted([p for p in imgdir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
print("generated images:", len(imgs))
if len(imgs) != 1000:
    raise SystemExit(f"Expected 1000 images, got {len(imgs)}")
print("first 5:", [p.name for p in imgs[:5]])
print("last 5:", [p.name for p in imgs[-5:]])
PY

# -------------------------
# meta.txt 저장
# -------------------------
cat > "$OUTDIR/meta.txt" <<META
candidate=$CAND
checkpoint=$CKPT
kimg=1843
mix=700x0.90 + 300x0.95
seeds_0_699=psi0.90
seeds_700_999=psi0.95
META

# -------------------------
# zip 생성 (이미지만 루트에)
# zip 명령어 안 쓰고 python zipfile 사용
# -------------------------
python3 - <<PY
from pathlib import Path
import zipfile

imgdir = Path("$IMGDIR")
zpath = Path("$OUTDIR/$CAND.zip")

imgs = sorted([p for p in imgdir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
if len(imgs) != 1000:
    raise SystemExit(f"Expected 1000 images before zipping, got {len(imgs)}")

with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
    for p in imgs:
        z.write(p, arcname=p.name)

print("zip created:", zpath)
print("size MB:", round(zpath.stat().st_size / 1024 / 1024, 2))
PY

# -------------------------
# zip 검증
# -------------------------
python3 - <<PY
import zipfile
from pathlib import Path

p = Path("$OUTDIR/$CAND.zip")
with zipfile.ZipFile(p, "r") as z:
    names = z.namelist()
    imgs = [n for n in names if n.lower().endswith((".png",".jpg",".jpeg"))]
    nested = [n for n in imgs if "/" in n.strip("/")]
    non_imgs = [n for n in names if not n.lower().endswith((".png",".jpg",".jpeg"))]

print("zip:", p)
print("images:", len(imgs))
print("nested:", len(nested))
print("non_imgs:", len(non_imgs))
print("size MB:", round(p.stat().st_size / 1024 / 1024, 2))

if len(imgs) != 1000:
    raise SystemExit(f"Zip does not contain 1000 images: {len(imgs)}")
if nested:
    raise SystemExit(f"Zip contains nested paths: {nested[:5]}")
if non_imgs:
    raise SystemExit(f"Zip contains non-image files: {non_imgs[:5]}")
if p.stat().st_size >= 200 * 1024 * 1024:
    raise SystemExit("Zip exceeds 200MB limit")

print("ZIP VALIDATION OK")
PY

echo "Done: $(date)"
