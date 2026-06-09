#!/bin/bash
#SBATCH --job-name=dgm_1f1843_m8095
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

RUN=/data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000/00000-celebvhq_rawframe_1f_p50_center256_jpg-mirror-paper256-gamma2-kimg2000-batch128-ada-target0.6-color-resumecustom
CKPT=$RUN/network-snapshot-001843.pkl

ROOT=/data/chsong/dgm-face/submissions/20260602_1fp50_k1843_mix800_090_200_095
CAND=k1843_mix800_090_200_095
OUTDIR=$ROOT/$CAND

TMP090=$OUTDIR/tmp_090
TMP095=$OUTDIR/tmp_095
IMGDIR=$OUTDIR/images

mkdir -p "$TMP090" "$TMP095" "$IMGDIR"

echo "Start: $(date)"
echo "Checkpoint: $CKPT"
test -f "$CKPT"

# 800 images: psi=0.90, seeds 0-799
python generate.py \
  --network="$CKPT" \
  --outdir="$TMP090" \
  --seeds=0-799 \
  --trunc=0.90

# 200 images: psi=0.95, seeds 800-999
python generate.py \
  --network="$CKPT" \
  --outdir="$TMP095" \
  --seeds=800-999 \
  --trunc=0.95

# Merge and rename to img_0000.png ... img_0999.png
python - <<'PY'
from pathlib import Path
import shutil

outdir = Path("/data/chsong/dgm-face/submissions/20260602_1fp50_k1843_mix800_090_200_095/k1843_mix800_090_200_095")
tmp090 = outdir / "tmp_090"
tmp095 = outdir / "tmp_095"
imgdir = outdir / "images"
imgdir.mkdir(exist_ok=True)

# first 800 from psi=0.90
for i, p in enumerate(sorted(tmp090.glob("seed*.png"))):
    dst = imgdir / f"img_{i:04d}.png"
    shutil.copy2(p, dst)

# next 200 from psi=0.95
offset = 800
for j, p in enumerate(sorted(tmp095.glob("seed*.png"))):
    dst = imgdir / f"img_{offset+j:04d}.png"
    shutil.copy2(p, dst)

imgs = sorted(imgdir.glob("*.png"))
print("generated images:", len(imgs))
if len(imgs) != 1000:
    raise SystemExit(f"Expected 1000 images, got {len(imgs)}")
print("first 5:", [x.name for x in imgs[:5]])
print("last 5:", [x.name for x in imgs[-5:]])
PY

# metadata
cat > "$OUTDIR/meta.txt" <<META
candidate=$CAND
checkpoint=$CKPT
kimg=1843
mix_rule=800 images with psi=0.90 (seeds 0-799) + 200 images with psi=0.95 (seeds 800-999)
META

# zip: images only at root, no nested folders
cd "$IMGDIR"
zip -q "$OUTDIR/${CAND}.zip" img_*.png
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

# zip validation
python - <<'PY'
import zipfile
from pathlib import Path

zpath = Path("/data/chsong/dgm-face/submissions/20260602_1fp50_k1843_mix800_090_200_095/k1843_mix800_090_200_095/k1843_mix800_090_200_095.zip")
with zipfile.ZipFile(zpath, "r") as z:
    imgs = [n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    nested = [n for n in imgs if "/" in n.strip("/")]
print("zip:", zpath)
print("image count:", len(imgs))
print("nested count:", len(nested))
print("size MB:", round(zpath.stat().st_size / 1024 / 1024, 2))
if len(imgs) != 1000:
    raise SystemExit(f"Zip does not contain 1000 images: {len(imgs)}")
if nested:
    raise SystemExit(f"Zip contains nested paths: {nested[:5]}")
PY

echo "Done: $(date)"
