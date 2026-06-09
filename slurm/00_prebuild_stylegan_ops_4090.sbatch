#!/bin/bash
#SBATCH --job-name=sg2op_build
#SBATCH --partition=R4090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face/stylegan2-ada-pytorch

export TORCH_CUDA_ARCH_LIST="8.9"
export TORCH_EXTENSIONS_DIR=/data/chsong/torch_extensions_stylegan2ada_py39_cu118_sm89
export MAX_JOBS=8

mkdir -p "$TORCH_EXTENSIONS_DIR"

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

print("upfirdn2d output:", y.shape)
print("PREBUILD_OK")
PY

find "$TORCH_EXTENSIONS_DIR" -type f | sort
