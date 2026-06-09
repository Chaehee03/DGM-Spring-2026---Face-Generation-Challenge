#!/bin/bash
#SBATCH --job-name=dgm3f_55_5868
#SBATCH --partition=R4090
#SBATCH --nodelist=vegi-01
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail

cd /workspace/chsong/dgm-face
source /workspace/chsong/miniconda3/bin/activate stylegan

export TORCH_CUDA_ARCH_LIST="8.9"
export TORCH_EXTENSIONS_DIR=/data/chsong/torch_extensions_stylegan2ada_py39_cu118_sm89
mkdir -p "$TORCH_EXTENSIONS_DIR" /workspace/chsong/dgm-face/logs

CKPT5500="/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k5500-from2000/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg3500-batch128-ada-target0.6-color-resumecustom/network-snapshot-003500.pkl"
CKPT5868="/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k6500-from5500/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg1000-batch128-ada-target0.6-color-resumecustom/network-snapshot-000368.pkl"

ROOT="/data/chsong/dgm-face/submissions/20260609_3f_k5500_875_k5868_125_mix750_090_250_095"
CAND="3f_k5500_875_k5868_125_mix750_090_250_095"
WORK="$ROOT/_work"

mkdir -p "$ROOT" "$WORK"

echo "Start: $(date)"
echo "CKPT5500=$CKPT5500"
echo "CKPT5868=$CKPT5868"

ls -lh "$CKPT5500"
ls -lh "$CKPT5868"

make_seed_csv () {
  python - "$1" "$2" <<'PY'
import sys
s = int(sys.argv[1])
e = int(sys.argv[2])
print(",".join(str(i) for i in range(s, e + 1)))
PY
}

gen_group () {
  local gpu="$1"
  local ckpt="$2"
  local trunc="$3"
  local s="$4"
  local e="$5"
  local outdir="$6"

  mkdir -p "$outdir"
  local seeds
  seeds=$(make_seed_csv "$s" "$e")

  echo "GPU=$gpu CKPT=$ckpt TRUNC=$trunc SEEDS=$s-$e OUT=$outdir"

  CUDA_VISIBLE_DEVICES="$gpu" python stylegan2-ada-pytorch/generate.py \
    --network="$ckpt" \
    --outdir="$outdir" \
    --seeds="$seeds" \
    --trunc="$trunc" \
    --noise-mode=const
}

# 구성
# k5500: 875장 = 656 @0.90 + 219 @0.95
# k5868: 125장 =  94 @0.90 +  31 @0.95
# seed는 서로 안 겹치게 배정
gen_group 0 "$CKPT5500" 0.90 0   655 "$WORK/5500_p090" &
gen_group 1 "$CKPT5500" 0.95 656 874 "$WORK/5500_p095" &
gen_group 2 "$CKPT5868" 0.90 875 968 "$WORK/5868_p090" &
gen_group 3 "$CKPT5868" 0.95 969 999 "$WORK/5868_p095" &

wait

echo "Generation finished: $(date)"

export ROOT CAND WORK
python - <<'PY'
import os, shutil, zipfile
from pathlib import Path

root = Path(os.environ["ROOT"])
cand = os.environ["CAND"]
work = Path(os.environ["WORK"])

final_dir = root / cand
final_dir.mkdir(parents=True, exist_ok=True)

groups = [
    "5500_p090",
    "5500_p095",
    "5868_p090",
    "5868_p095",
]

idx = 0
for g in groups:
    gdir = work / g
    imgs = sorted(gdir.glob("*.png"))
    print(g, len(imgs))
    for src in imgs:
        dst = final_dir / f"img_{idx:04d}.png"
        shutil.copy2(src, dst)
        idx += 1

print("copied images:", idx)
assert idx == 1000, f"Expected 1000 images, got {idx}"

zpath = final_dir / f"{cand}.zip"
with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for img in sorted(final_dir.glob("img_*.png")):
        zf.write(img, arcname=img.name)

with zipfile.ZipFile(zpath, "r") as zf:
    names = zf.namelist()
    imgs = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    nested = [n for n in imgs if "/" in n.strip("/")]
    non_imgs = [n for n in names if not n.lower().endswith((".png", ".jpg", ".jpeg"))]

size_mb = zpath.stat().st_size / 1024 / 1024

print("ZIP:", zpath)
print("images:", len(imgs))
print("nested:", len(nested))
print("non_imgs:", len(non_imgs))
print("size MB:", round(size_mb, 2))

ok = (len(imgs) == 1000) and (len(nested) == 0) and (len(non_imgs) == 0) and (size_mb < 200)
print("OK" if ok else "BAD")
PY

echo "Done: $(date)"
