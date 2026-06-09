#!/bin/bash
#SBATCH --job-name=dgm_ens55_5868
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

CKPT_5500=/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k5500-from2000/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg3500-batch128-ada-target0.6-color-resumecustom/network-snapshot-003500.pkl

CKPT_5868=/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k6500-from5500/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg1000-batch128-ada-target0.6-color-resumecustom/network-snapshot-000368.pkl

for p in "$CKPT_5500" "$CKPT_5868"; do
  if [ ! -f "$p" ]; then
    echo "ERROR: checkpoint not found: $p"
    exit 1
  fi
done

ROOT=/data/chsong/dgm-face/submissions/20260608_3f_k5500_900_k5868_100_mix750_090_250_095
CAND=3f_k5500_900_k5868_100_mix750_090_250_095
OUTDIR=$ROOT/$CAND
IMGDIR=$OUTDIR/images

rm -rf "$OUTDIR"
mkdir -p "$IMGDIR"

echo "Start: $(date)"
echo "CKPT_5500=$CKPT_5500"
echo "CKPT_5868=$CKPT_5868"
echo "OUTDIR=$OUTDIR"

# Total:
# 750 @ psi=0.90 = 675 from k5500 + 75 from k5868
# 250 @ psi=0.95 = 225 from k5500 + 25 from k5868

# k5500, psi=0.90, 675 images: seeds 0-674
CUDA_VISIBLE_DEVICES=0 python generate.py --network="$CKPT_5500" --seeds=0-168   --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k5500_p090" &
P1=$!
CUDA_VISIBLE_DEVICES=1 python generate.py --network="$CKPT_5500" --seeds=169-337 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k5500_p090" &
P2=$!
CUDA_VISIBLE_DEVICES=2 python generate.py --network="$CKPT_5500" --seeds=338-506 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k5500_p090" &
P3=$!
CUDA_VISIBLE_DEVICES=3 python generate.py --network="$CKPT_5500" --seeds=507-674 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k5500_p090" &
P4=$!

# k5500, psi=0.95, 225 images: seeds 675-899
CUDA_VISIBLE_DEVICES=4 python generate.py --network="$CKPT_5500" --seeds=675-730 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k5500_p095" &
P5=$!
CUDA_VISIBLE_DEVICES=5 python generate.py --network="$CKPT_5500" --seeds=731-786 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k5500_p095" &
P6=$!
CUDA_VISIBLE_DEVICES=6 python generate.py --network="$CKPT_5500" --seeds=787-842 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k5500_p095" &
P7=$!
CUDA_VISIBLE_DEVICES=7 python generate.py --network="$CKPT_5500" --seeds=843-899 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k5500_p095" &
P8=$!

wait $P1 $P2 $P3 $P4 $P5 $P6 $P7 $P8

# k5868, 100 images total
CUDA_VISIBLE_DEVICES=0 python generate.py --network="$CKPT_5868" --seeds=900-974 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k5868_p090" &
P9=$!
CUDA_VISIBLE_DEVICES=1 python generate.py --network="$CKPT_5868" --seeds=975-999 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k5868_p095" &
P10=$!

wait $P9 $P10

echo "Generation finished: $(date)"

python3 - <<PY
from pathlib import Path
import zipfile, re, shutil

cand = "$CAND"
outdir = Path("$OUTDIR")
imgdir = Path("$IMGDIR")
flatdir = outdir / "flat"
zpath = outdir / f"{cand}.zip"

if flatdir.exists():
    shutil.rmtree(flatdir)
flatdir.mkdir(parents=True, exist_ok=True)

groups = [
    ("k5500_p090", 675),
    ("k5500_p095", 225),
    ("k5868_p090", 75),
    ("k5868_p095", 25),
]

def seed_key(p):
    m = re.search(r"(\\d+)", p.stem)
    return int(m.group(1)) if m else 10**18

all_imgs = []
for group, expected in groups:
    d = imgdir / group
    imgs = sorted(
        [p for p in d.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]],
        key=seed_key,
    )
    print(group, len(imgs))
    if len(imgs) != expected:
        raise SystemExit(f"{group}: expected {expected}, got {len(imgs)}")
    all_imgs.extend(imgs)

if len(all_imgs) != 1000:
    raise SystemExit(f"Expected 1000 total images, got {len(all_imgs)}")

for i, p in enumerate(all_imgs):
    shutil.copy2(p, flatdir / f"img_{i:04d}.png")

meta = outdir / "meta.txt"
meta.write_text(
    f"candidate={cand}\\n"
    f"source=checkpoint ensemble\\n"
    f"k5500_checkpoint=$CKPT_5500\\n"
    f"k5868_checkpoint=$CKPT_5868\\n"
    f"composition=900 images from k5500 + 100 images from k5868\\n"
    f"overall_mix=750@psi0.90 + 250@psi0.95\\n"
    f"k5500=675@psi0.90 seeds0-674 + 225@psi0.95 seeds675-899\\n"
    f"k5868=75@psi0.90 seeds900-974 + 25@psi0.95 seeds975-999\\n"
    f"noise_mode=const\\n"
)

if zpath.exists():
    zpath.unlink()

imgs = sorted(flatdir.glob("img_*.png"))

with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
    for p in imgs:
        z.write(p, arcname=p.name)

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
