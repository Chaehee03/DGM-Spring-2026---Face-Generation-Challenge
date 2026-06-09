#!/bin/bash
#SBATCH --job-name=dgm_next4
#SBATCH --partition=A100
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-04
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-06:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

# =========================
# Checkpoints
# =========================

CKPT_1F=$(find \
  /data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000 \
  /data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000-cont \
  -name "network-snapshot-001843.pkl" 2>/dev/null | sort | tail -n 1)

# 3f k5500 = from 3f k2000, additional 3500 kimg.
# Continuation snapshot may be named network-snapshot-003500.pkl.
CKPT_3F=$(find \
  /data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k5500-from2000 \
  -name "network-snapshot-003500.pkl" 2>/dev/null | sort | tail -n 1)

# fallback: take latest snapshot in the k5500 continuation folder
if [ -z "${CKPT_3F}" ]; then
  CKPT_3F=$(find \
    /data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k5500-from2000 \
    -name "network-snapshot-*.pkl" 2>/dev/null | sort | tail -n 1)
fi

echo "CKPT_1F=$CKPT_1F"
echo "CKPT_3F=$CKPT_3F"

test -f "$CKPT_1F"
test -f "$CKPT_3F"

if [[ "$CKPT_1F" != *"network-snapshot-001843.pkl" ]]; then
  echo "ERROR: wrong 1f checkpoint: $CKPT_1F"
  exit 1
fi

echo "Using 1f checkpoint: $CKPT_1F"
echo "Using 3f checkpoint: $CKPT_3F"

nvidia-smi

# =========================
# Output dirs
# =========================

ROOT=/data/chsong/dgm-face/submissions/20260602_next4_metric_candidates
rm -rf "$ROOT"

C1=$ROOT/1f_k1843_mix500_075_500_090
C2=$ROOT/1f_k1843_mix600_075_400_090
C3=$ROOT/1f_k1843_mix700_075_300_095
C4=$ROOT/3f_k5500_mix700_075_300_090

mkdir -p "$C1/images" "$C2/images" "$C3/images" "$C4/images"

echo "ROOT=$ROOT"
echo "Start generation: $(date)"

# =========================
# Candidate 1
# 1f k1843: 500 @ 0.75 + 500 @ 0.90
# =========================
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --network="$CKPT_1F" \
  --seeds=0-499 \
  --trunc=0.75 \
  --noise-mode=const \
  --outdir="$C1/images" &

CUDA_VISIBLE_DEVICES=1 python generate.py \
  --network="$CKPT_1F" \
  --seeds=500-999 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$C1/images" &

# =========================
# Candidate 2
# 1f k1843: 600 @ 0.75 + 400 @ 0.90
# =========================
CUDA_VISIBLE_DEVICES=2 python generate.py \
  --network="$CKPT_1F" \
  --seeds=0-599 \
  --trunc=0.75 \
  --noise-mode=const \
  --outdir="$C2/images" &

CUDA_VISIBLE_DEVICES=3 python generate.py \
  --network="$CKPT_1F" \
  --seeds=600-999 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$C2/images" &

# =========================
# Candidate 3
# 1f k1843: 700 @ 0.75 + 300 @ 0.95
# =========================
CUDA_VISIBLE_DEVICES=4 python generate.py \
  --network="$CKPT_1F" \
  --seeds=0-699 \
  --trunc=0.75 \
  --noise-mode=const \
  --outdir="$C3/images" &

CUDA_VISIBLE_DEVICES=5 python generate.py \
  --network="$CKPT_1F" \
  --seeds=700-999 \
  --trunc=0.95 \
  --noise-mode=const \
  --outdir="$C3/images" &

# =========================
# Candidate 4
# 3f effective k5500: 700 @ 0.75 + 300 @ 0.90
# =========================
CUDA_VISIBLE_DEVICES=6 python generate.py \
  --network="$CKPT_3F" \
  --seeds=0-699 \
  --trunc=0.75 \
  --noise-mode=const \
  --outdir="$C4/images" &

CUDA_VISIBLE_DEVICES=7 python generate.py \
  --network="$CKPT_3F" \
  --seeds=700-999 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$C4/images" &

wait

echo "Generation done: $(date)"

# =========================
# Zip and verify
# =========================

python - <<PY
from pathlib import Path
import zipfile
import re

root = Path("$ROOT")

candidates = [
    {
        "name": "1f_k1843_mix500_075_500_090",
        "dir": Path("$C1"),
        "checkpoint": "$CKPT_1F",
        "frame": "1f_p50",
        "effective_kimg": "1843",
        "mix": "500@psi0.75 + 500@psi0.90",
    },
    {
        "name": "1f_k1843_mix600_075_400_090",
        "dir": Path("$C2"),
        "checkpoint": "$CKPT_1F",
        "frame": "1f_p50",
        "effective_kimg": "1843",
        "mix": "600@psi0.75 + 400@psi0.90",
    },
    {
        "name": "1f_k1843_mix700_075_300_095",
        "dir": Path("$C3"),
        "checkpoint": "$CKPT_1F",
        "frame": "1f_p50",
        "effective_kimg": "1843",
        "mix": "700@psi0.75 + 300@psi0.95",
    },
    {
        "name": "3f_k5500_mix700_075_300_090",
        "dir": Path("$C4"),
        "checkpoint": "$CKPT_3F",
        "frame": "3f_p20_p50_p80",
        "effective_kimg": "5500",
        "mix": "700@psi0.75 + 300@psi0.90",
    },
]

def seed_key(p):
    m = re.search(r"(\\d+)", p.stem)
    return int(m.group(1)) if m else 10**18

for c in candidates:
    name = c["name"]
    outdir = c["dir"]
    imgdir = outdir / "images"

    imgs = sorted(imgdir.glob("*.png"), key=seed_key)
    print()
    print("candidate:", name)
    print("image_count:", len(imgs))

    if len(imgs) != 1000:
        raise SystemExit(f"{name}: expected 1000 images, got {len(imgs)}")

    meta = outdir / "meta.txt"
    meta.write_text(
        f"candidate={name}\\n"
        f"frame={c['frame']}\\n"
        f"effective_kimg={c['effective_kimg']}\\n"
        f"checkpoint={c['checkpoint']}\\n"
        f"mix={c['mix']}\\n"
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
        raise SystemExit(f"{name}: zip image count is not 1000")
    if nested:
        raise SystemExit(f"{name}: zip has nested paths")
    if non_imgs:
        raise SystemExit(f"{name}: zip has non-image files")
    if size_mb >= 200:
        raise SystemExit(f"{name}: zip exceeds 200MB")

print()
print("ALL_ZIP_OK")
PY

echo "All done: $(date)"
echo "Output root: $ROOT"

find "$ROOT" -maxdepth 2 -type f \( -name "*.zip" -o -name "meta.txt" \) -print | sort
