#!/bin/bash
#SBATCH --job-name=dgm3f_k6500
#SBATCH --partition=R4090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=1-12:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

# RTX 4090 = compute capability 8.9
# Use the prebuilt custom CUDA op cache.
export TORCH_CUDA_ARCH_LIST="8.9"
export TORCH_EXTENSIONS_DIR=/data/chsong/torch_extensions_stylegan2ada_py39_cu118_sm89
export MAX_JOBS=8
mkdir -p "$TORCH_EXTENSIONS_DIR"

DATA=/data/chsong/dgm-face/datasets/celebvhq_rawframe_3f_0205080_center256_jpg.zip

CKPT_5500=/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k5500-from2000/00000-celebvhq_rawframe_3f_0205080_center256_jpg-mirror-paper256-gamma2-kimg3500-batch128-ada-target0.6-color-resumecustom/network-snapshot-003500.pkl

OUTDIR=/data/chsong/dgm-face/training-runs-rawframe-3f-ffhq-ada-color-k6500-from5500

echo "Start: $(date)"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-none}"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "DATA=$DATA"
echo "RESUME=$CKPT_5500"
echo "OUTDIR=$OUTDIR"

if [ ! -f "$DATA" ]; then
  echo "ERROR: dataset not found: $DATA"
  exit 1
fi

if [ ! -f "$CKPT_5500" ]; then
  echo "ERROR: checkpoint not found: $CKPT_5500"
  exit 1
fi

if [ ! -f "$TORCH_EXTENSIONS_DIR/upfirdn2d_plugin/upfirdn2d_plugin.so" ]; then
  echo "ERROR: prebuilt upfirdn2d_plugin.so not found"
  find "$TORCH_EXTENSIONS_DIR" -type f | sort || true
  exit 1
fi

nvidia-smi || true

# Precheck: this should load the prebuilt plugin before distributed training starts.
python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("TORCH_EXTENSIONS_DIR:", os.environ.get("TORCH_EXTENSIONS_DIR"))
print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))

from torch_utils.ops import upfirdn2d
x = torch.randn([1, 1, 16, 16], device="cuda")
f = torch.ones([1, 1], device="cuda")
y = upfirdn2d.upfirdn2d(x, f)
torch.cuda.synchronize()
print("UPFIRDN2D_PRECHECK_OK", y.shape)
PY

python train.py \
  --outdir="$OUTDIR" \
  --data="$DATA" \
  --resume="$CKPT_5500" \
  --gpus=8 \
  --cfg=paper256 \
  --gamma=2 \
  --batch=128 \
  --mirror=1 \
  --aug=ada \
  --target=0.6 \
  --augpipe=color \
  --snap=10 \
  --kimg=1000 \
  --metrics=none \
  --workers=8

echo "Done: $(date)"
echo "Final snapshots:"
find "$OUTDIR" -type f -name "network-snapshot-*.pkl" -print | sort
