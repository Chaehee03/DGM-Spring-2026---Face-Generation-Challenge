#!/bin/bash
#SBATCH --job-name=dgm_3f55_m7595
#SBATCH --partition=R4090
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

: "${CKPT:?You must pass CKPT=/path/to/network-snapshot-003500.pkl}"

DATE_TAG=20260607
CAND=3f_k5500_mix750_090_250_095
ROOT=/data/chsong/dgm-face/submissions/${DATE_TAG}_${CAND}
OUTDIR=${ROOT}/${CAND}
IMGDIR=${OUTDIR}/images
ZIPROOT=${OUTDIR}/zip_root

mkdir -p "$IMGDIR" "$ZIPROOT"

echo "Start: $(date)"
echo "Checkpoint: $CKPT"
echo "Candidate: $CAND"
echo "OUTDIR: $OUTDIR"

run_gen () {
    GPU_ID=$1
    SEEDS=$2
    TRUNC=$3

    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    export TORCH_EXTENSIONS_DIR=/tmp/${USER}/torch_ext_${SLURM_JOB_ID}_gpu${GPU_ID}
    mkdir -p "${TORCH_EXTENSIONS_DIR}"

    echo "[GPU ${GPU_ID}] seeds=${SEEDS}, trunc=${TRUNC}"
    python generate.py \
        --network="$CKPT" \
        --outdir="$IMGDIR" \
        --seeds="$SEEDS" \
        --trunc="$TRUNC"
}

# 1000장 = 8개 GPU에 125장씩 균등 분배
# 0-749   -> psi 0.90
# 750-999 -> psi 0.95

run_gen 0 0-124     0.90 &
run_gen 1 125-249   0.90 &
run_gen 2 250-374   0.90 &
run_gen 3 375-499   0.90 &
run_gen 4 500-624   0.90 &
run_gen 5 625-749   0.90 &
run_gen 6 750-874   0.95 &
run_gen 7 875-999   0.95 &

wait

echo "Generation finished: $(date)"

python - <<PY
from pathlib import Path
imgdir = Path("${IMGDIR}")
imgs = sorted(imgdir.glob("seed*.png"))
print("generated images:", len(imgs))
if len(imgs) != 1000:
    raise SystemExit(f"Expected 1000 images, got {len(imgs)}")
PY

# 제출용 zip 루트에 img_0000.png ~ img_0999.png 형태로 정리
python - <<PY
from pathlib import Path
import shutil
imgdir = Path("${IMGDIR}")
ziproot = Path("${ZIPROOT}")

for p in ziproot.glob("*"):
    if p.is_file():
        p.unlink()

imgs = sorted(imgdir.glob("seed*.png"))
for i, src in enumerate(imgs):
    dst = ziproot / f"img_{i:04d}.png"
    shutil.copy2(src, dst)

print("prepared zip_root images:", len(list(ziproot.glob("img_*.png"))))
PY

# zip 생성 (이미지만 루트에)
python - <<PY
from pathlib import Path
import zipfile

outdir = Path("${OUTDIR}")
ziproot = Path("${ZIPROOT}")
zpath = outdir / "${CAND}.zip"

if zpath.exists():
    zpath.unlink()

with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
    for p in sorted(ziproot.glob("img_*.png")):
        z.write(p, arcname=p.name)

print("zip created:", zpath)
print("size MB:", round(zpath.stat().st_size / 1024 / 1024, 2))
PY

# 메타 정보 (zip 바깥)
cat > "${OUTDIR}/meta.txt" <<META
candidate=${CAND}
checkpoint=${CKPT}
effective_total_kimg=5500
mix=750@0.90 + 250@0.95
seed_range_1=0-749
seed_range_2=750-999
META

# zip 검증
python - <<PY
from pathlib import Path
import zipfile

zpath = Path("${OUTDIR}/${CAND}.zip")
with zipfile.ZipFile(zpath, "r") as z:
    names = z.namelist()
    imgs = [n for n in names if n.lower().endswith((".png",".jpg",".jpeg"))]
    nested = [n for n in imgs if "/" in n.strip("/")]
    non_imgs = [n for n in names if not n.lower().endswith((".png",".jpg",".jpeg"))]

print("zip:", zpath)
print("images:", len(imgs))
print("nested:", len(nested))
print("non_imgs:", len(non_imgs))
print("size MB:", round(zpath.stat().st_size / 1024 / 1024, 2))

if len(imgs) != 1000:
    raise SystemExit(f"Zip image count != 1000: {len(imgs)}")
if nested:
    raise SystemExit(f"Nested files found in zip: {nested[:5]}")
if non_imgs:
    raise SystemExit(f"Non-image files found in zip: {non_imgs[:5]}")
if zpath.stat().st_size >= 200 * 1024 * 1024:
    raise SystemExit("Zip exceeds 200 MB")

print("ZIP VALIDATION: OK")
PY

echo "Done: $(date)"
