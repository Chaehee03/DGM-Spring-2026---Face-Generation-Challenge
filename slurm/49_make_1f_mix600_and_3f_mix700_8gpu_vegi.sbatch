#!/bin/bash
#SBATCH --job-name=dgm_2subs_8gpu
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

CKPT_1F=/data/chsong/dgm-face/training-runs-rawframe-1f-p50-ffhq-ada-color-k2000/00000-celebvhq_rawframe_1f_p50_center256_jpg-mirror-paper256-gamma2-kimg2000-batch128-ada-target0.6-color-resumecustom/network-snapshot-001843.pkl

CKPT_3F=/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k5500-from2000/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg3500-batch128-ada-target0.6-color-resumecustom/network-snapshot-003500.pkl

ROOT_1F=/data/chsong/dgm-face/submissions/20260603_1fp50_k1843_mix600_090_400_095
CAND_1F=1f_k1843_mix600_090_400_095
OUT_1F=$ROOT_1F/$CAND_1F
IMG_1F=$OUT_1F/images

ROOT_3F=/data/chsong/dgm-face/submissions/20260603_3f_k5500_mix700_090_300_095
CAND_3F=3f_k5500_mix700_090_300_095
OUT_3F=$ROOT_3F/$CAND_3F
IMG_3F=$OUT_3F/images

rm -rf "$OUT_1F" "$OUT_3F"
mkdir -p "$IMG_1F" "$IMG_3F"

echo "Start: $(date)"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-none}"
echo "CKPT_1F=$CKPT_1F"
echo "CKPT_3F=$CKPT_3F"
echo "OUT_1F=$OUT_1F"
echo "OUT_3F=$OUT_3F"

if [ ! -f "$CKPT_1F" ]; then
  echo "ERROR: missing 1f checkpoint: $CKPT_1F"
  exit 1
fi

if [ ! -f "$CKPT_3F" ]; then
  echo "ERROR: missing 3f checkpoint: $CKPT_3F"
  exit 1
fi

nvidia-smi || true

echo "[1/8] 1f psi0.90 seeds 0-299 on GPU0"
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --network="$CKPT_1F" \
  --seeds=0-299 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$IMG_1F" &
P1=$!

echo "[2/8] 1f psi0.90 seeds 300-599 on GPU1"
CUDA_VISIBLE_DEVICES=1 python generate.py \
  --network="$CKPT_1F" \
  --seeds=300-599 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$IMG_1F" &
P2=$!

echo "[3/8] 1f psi0.95 seeds 600-799 on GPU2"
CUDA_VISIBLE_DEVICES=2 python generate.py \
  --network="$CKPT_1F" \
  --seeds=600-799 \
  --trunc=0.95 \
  --noise-mode=const \
  --outdir="$IMG_1F" &
P3=$!

echo "[4/8] 1f psi0.95 seeds 800-999 on GPU3"
CUDA_VISIBLE_DEVICES=3 python generate.py \
  --network="$CKPT_1F" \
  --seeds=800-999 \
  --trunc=0.95 \
  --noise-mode=const \
  --outdir="$IMG_1F" &
P4=$!

echo "[5/8] 3f psi0.90 seeds 0-349 on GPU4"
CUDA_VISIBLE_DEVICES=4 python generate.py \
  --network="$CKPT_3F" \
  --seeds=0-349 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$IMG_3F" &
P5=$!

echo "[6/8] 3f psi0.90 seeds 350-699 on GPU5"
CUDA_VISIBLE_DEVICES=5 python generate.py \
  --network="$CKPT_3F" \
  --seeds=350-699 \
  --trunc=0.90 \
  --noise-mode=const \
  --outdir="$IMG_3F" &
P6=$!

echo "[7/8] 3f psi0.95 seeds 700-849 on GPU6"
CUDA_VISIBLE_DEVICES=6 python generate.py \
  --network="$CKPT_3F" \
  --seeds=700-849 \
  --trunc=0.95 \
  --noise-mode=const \
  --outdir="$IMG_3F" &
P7=$!

echo "[8/8] 3f psi0.95 seeds 850-999 on GPU7"
CUDA_VISIBLE_DEVICES=7 python generate.py \
  --network="$CKPT_3F" \
  --seeds=850-999 \
  --trunc=0.95 \
  --noise-mode=const \
  --outdir="$IMG_3F" &
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

jobs = [
    {
        "cand": "$CAND_1F",
        "out": Path("$OUT_1F"),
        "img": Path("$IMG_1F"),
        "meta": {
            "candidate": "$CAND_1F",
            "source": "1f_p50_rawframe",
            "checkpoint": "$CKPT_1F",
            "effective_kimg": "1843",
            "mix": "600@psi0.90 + 400@psi0.95",
            "seeds": "0-599@0.90,600-999@0.95",
            "noise_mode": "const",
        },
    },
    {
        "cand": "$CAND_3F",
        "out": Path("$OUT_3F"),
        "img": Path("$IMG_3F"),
        "meta": {
            "candidate": "$CAND_3F",
            "source": "3f_rawframe_0205080",
            "checkpoint": "$CKPT_3F",
            "effective_kimg": "5500",
            "checkpoint_file_kimg": "3500_cont_from_2000",
            "mix": "700@psi0.90 + 300@psi0.95",
            "seeds": "0-699@0.90,700-999@0.95",
            "noise_mode": "const",
        },
    },
]

def seed_key(p):
    m = re.search(r"(\\d+)", p.stem)
    return int(m.group(1)) if m else 10**18

for j in jobs:
    cand = j["cand"]
    outdir = j["out"]
    imgdir = j["img"]
    zpath = outdir / f"{cand}.zip"

    imgs = sorted(
        [p for p in imgdir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]],
        key=seed_key,
    )

    print("\\n===", cand, "===")
    print("generated images:", len(imgs))

    if len(imgs) != 1000:
        raise SystemExit(f"{cand}: expected 1000 images, got {len(imgs)}")

    meta_text = "\\n".join([f"{k}={v}" for k, v in j["meta"].items()]) + "\\n"
    (outdir / "meta.txt").write_text(meta_text)

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
        raise SystemExit(f"{cand}: zip image count is not 1000")
    if nested:
        raise SystemExit(f"{cand}: zip has nested paths")
    if non_imgs:
        raise SystemExit(f"{cand}: zip has non-image files")
    if size_mb >= 200:
        raise SystemExit(f"{cand}: zip exceeds 200MB")

    print("ZIP_VALIDATION_OK")

PY

echo "Done: $(date)"
echo "Final files:"
find "$ROOT_1F" "$ROOT_3F" -maxdepth 3 -type f \( -name "*.zip" -o -name "meta.txt" \) -print | sort
