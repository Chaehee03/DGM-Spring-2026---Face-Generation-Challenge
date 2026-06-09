#!/bin/bash
#SBATCH --job-name=dgm_55_5868_6114
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

CKPT_6114=/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k6500-from5500/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg1000-batch128-ada-target0.6-color-resumecustom/network-snapshot-000614.pkl

for p in "$CKPT_5500" "$CKPT_5868" "$CKPT_6114"; do
  if [ ! -f "$p" ]; then
    echo "ERROR: checkpoint not found: $p"
    exit 1
  fi
done

ROOT=/data/chsong/dgm-face/submissions/20260609_3f_k5500_850_k5868_100_k6114_50_mix750_090_250_095
CAND=3f_k5500_850_k5868_100_k6114_50_mix750_090_250_095
OUTDIR=$ROOT/$CAND
IMGDIR=$OUTDIR/images
FLATDIR=$OUTDIR/flat

rm -rf "$OUTDIR"
mkdir -p "$IMGDIR" "$FLATDIR"

echo "Start: $(date)"
echo "HOST=$(hostname)"
echo "CKPT_5500=$CKPT_5500"
echo "CKPT_5868=$CKPT_5868"
echo "CKPT_6114=$CKPT_6114"
echo "OUTDIR=$OUTDIR"

nvidia-smi || true

# ---------------------------------------------------------------------
# Target composition:
# k5500: 850 images = 638 @ psi0.90 + 212 @ psi0.95
# k5868: 100 images = 75  @ psi0.90 + 25  @ psi0.95
# k6114: 50  images = 37  @ psi0.90 + 13  @ psi0.95
#
# Overall:
# psi0.90 = 638 + 75 + 37 = 750
# psi0.95 = 212 + 25 + 13 = 250
#
# Seed ranges are non-overlapping:
# k5500 p090: 0-637
# k5500 p095: 638-849
# k5868 p090: 850-924
# k5868 p095: 925-949
# k6114 p090: 950-986
# k6114 p095: 987-999
# ---------------------------------------------------------------------

# k5500, psi=0.90, 638 images
CUDA_VISIBLE_DEVICES=0 python generate.py --network="$CKPT_5500" --seeds=0-159   --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k5500_p090" &
P1=$!
CUDA_VISIBLE_DEVICES=1 python generate.py --network="$CKPT_5500" --seeds=160-319 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k5500_p090" &
P2=$!
CUDA_VISIBLE_DEVICES=2 python generate.py --network="$CKPT_5500" --seeds=320-478 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k5500_p090" &
P3=$!
CUDA_VISIBLE_DEVICES=3 python generate.py --network="$CKPT_5500" --seeds=479-637 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k5500_p090" &
P4=$!

# k5500, psi=0.95, 212 images
CUDA_VISIBLE_DEVICES=4 python generate.py --network="$CKPT_5500" --seeds=638-690 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k5500_p095" &
P5=$!
CUDA_VISIBLE_DEVICES=5 python generate.py --network="$CKPT_5500" --seeds=691-743 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k5500_p095" &
P6=$!
CUDA_VISIBLE_DEVICES=6 python generate.py --network="$CKPT_5500" --seeds=744-796 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k5500_p095" &
P7=$!
CUDA_VISIBLE_DEVICES=7 python generate.py --network="$CKPT_5500" --seeds=797-849 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k5500_p095" &
P8=$!

wait $P1 $P2 $P3 $P4 $P5 $P6 $P7 $P8

# k5868, 100 images
CUDA_VISIBLE_DEVICES=0 python generate.py --network="$CKPT_5868" --seeds=850-924 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k5868_p090" &
P9=$!
CUDA_VISIBLE_DEVICES=1 python generate.py --network="$CKPT_5868" --seeds=925-949 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k5868_p095" &
P10=$!

# k6114, 50 images
CUDA_VISIBLE_DEVICES=2 python generate.py --network="$CKPT_6114" --seeds=950-986 --trunc=0.90 --noise-mode=const --outdir="$IMGDIR/k6114_p090" &
P11=$!
CUDA_VISIBLE_DEVICES=3 python generate.py --network="$CKPT_6114" --seeds=987-999 --trunc=0.95 --noise-mode=const --outdir="$IMGDIR/k6114_p095" &
P12=$!

wait $P9 $P10 $P11 $P12

echo "Generation finished: $(date)"

python3 - <<PY
from pathlib import Path
import zipfile
import re
import shutil
import hashlib
import struct
from collections import Counter

cand = "$CAND"
outdir = Path("$OUTDIR")
imgdir = Path("$IMGDIR")
flatdir = Path("$FLATDIR")
zpath = outdir / f"{cand}.zip"

groups = [
    ("k5500_p090", 638),
    ("k5500_p095", 212),
    ("k5868_p090", 75),
    ("k5868_p095", 25),
    ("k6114_p090", 37),
    ("k6114_p095", 13),
]

def seed_key(p):
    m = re.search(r"(\\d+)", p.stem)
    return int(m.group(1)) if m else 10**18

def png_size(data):
    if data[:8] != b"\\x89PNG\\r\\n\\x1a\\n":
        return None
    if data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])

if flatdir.exists():
    shutil.rmtree(flatdir)
flatdir.mkdir(parents=True, exist_ok=True)

all_imgs = []
print("Group counts:")
for group, expected in groups:
    d = imgdir / group
    imgs = sorted(
        [p for p in d.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]],
        key=seed_key,
    )
    print(f"{group}: {len(imgs)}")
    if len(imgs) != expected:
        raise SystemExit(f"{group}: expected {expected}, got {len(imgs)}")
    all_imgs.extend(imgs)

if len(all_imgs) != 1000:
    raise SystemExit(f"Expected 1000 total images, got {len(all_imgs)}")

# Copy to flat submit directory with standardized names.
for i, p in enumerate(all_imgs):
    shutil.copy2(p, flatdir / f"img_{i:04d}.png")

flat_imgs = sorted(flatdir.glob("img_*.png"))
if len(flat_imgs) != 1000:
    raise SystemExit(f"Expected 1000 flat images, got {len(flat_imgs)}")

# Exact duplicate and PNG dimension check before zip.
hashes = Counter()
dims = Counter()
bad = []
for p in flat_imgs:
    try:
        data = p.read_bytes()
        hashes[hashlib.md5(data).hexdigest()] += 1
        sz = png_size(data)
        if sz is None:
            bad.append((p.name, "invalid png header"))
        else:
            dims[sz] += 1
    except Exception as e:
        bad.append((p.name, repr(e)))

dups = sum(c - 1 for c in hashes.values() if c > 1)

print("Pre-zip check:")
print("flat_images:", len(flat_imgs))
print("png_dimensions:", dims)
print("bad_images:", len(bad))
print("exact_duplicate_images:", dups)

if bad:
    print("bad examples:", bad[:5])
if len(flat_imgs) != 1000:
    raise SystemExit("flat image count is not 1000")
if bad:
    raise SystemExit("bad/corrupt images exist")
if dups != 0:
    raise SystemExit("exact duplicate images exist")
if dims != Counter({(256, 256): 1000}):
    raise SystemExit(f"unexpected image dimensions: {dims}")

meta = outdir / "meta.txt"
meta.write_text(
    f"candidate={cand}\\n"
    f"source=checkpoint ensemble\\n"
    f"k5500_checkpoint=$CKPT_5500\\n"
    f"k5868_checkpoint=$CKPT_5868\\n"
    f"k6114_checkpoint=$CKPT_6114\\n"
    f"composition=k5500:850,k5868:100,k6114:50\\n"
    f"overall_mix=750@psi0.90 + 250@psi0.95\\n"
    f"k5500=638@psi0.90 seeds0-637 + 212@psi0.95 seeds638-849\\n"
    f"k5868=75@psi0.90 seeds850-924 + 25@psi0.95 seeds925-949\\n"
    f"k6114=37@psi0.90 seeds950-986 + 13@psi0.95 seeds987-999\\n"
    f"noise_mode=const\\n"
    f"exact_duplicate_images=0\\n"
)

if zpath.exists():
    zpath.unlink()

with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
    for p in flat_imgs:
        z.write(p, arcname=p.name)

# Post-zip validation.
with zipfile.ZipFile(zpath, "r") as z:
    names = sorted(z.namelist())
    image_names = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    nested = [n for n in image_names if "/" in n.strip("/")]
    non_imgs = [n for n in names if not n.lower().endswith((".png", ".jpg", ".jpeg"))]

    z_hashes = Counter()
    z_dims = Counter()
    z_bad = []
    for n in image_names:
        try:
            data = z.read(n)
            z_hashes[hashlib.md5(data).hexdigest()] += 1
            sz = png_size(data)
            if sz is None:
                z_bad.append((n, "invalid png header"))
            else:
                z_dims[sz] += 1
        except Exception as e:
            z_bad.append((n, repr(e)))

z_dups = sum(c - 1 for c in z_hashes.values() if c > 1)
size_mb = zpath.stat().st_size / 1024 / 1024
expected_names = [f"img_{i:04d}.png" for i in range(1000)]

print("Post-zip check:")
print("zip:", zpath)
print("zip_size_MB:", round(size_mb, 2))
print("num_images:", len(image_names))
print("nested_paths:", len(nested))
print("non_image_files:", len(non_imgs))
print("png_dimensions:", z_dims)
print("corrupt_or_bad_images:", len(z_bad))
print("exact_duplicate_images:", z_dups)
print("filename_sequence_ok:", names == expected_names)

if len(image_names) != 1000:
    raise SystemExit("zip image count is not 1000")
if nested:
    raise SystemExit("zip has nested paths")
if non_imgs:
    raise SystemExit("zip has non-image files")
if z_bad:
    raise SystemExit("zip has bad images")
if z_dups != 0:
    raise SystemExit("zip has exact duplicate images")
if z_dims != Counter({(256, 256): 1000}):
    raise SystemExit(f"zip has unexpected dimensions: {z_dims}")
if names != expected_names:
    raise SystemExit("zip filenames are not img_0000.png ... img_0999.png")
if size_mb >= 200:
    raise SystemExit("zip exceeds 200MB")

print("ZIP_VALIDATION_OK")
PY

echo "Done: $(date)"
find "$ROOT" -maxdepth 3 -type f \( -name "*.zip" -o -name "meta.txt" \) -print -exec ls -lh {} \;
