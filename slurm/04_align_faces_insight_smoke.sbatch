#!/bin/bash
#SBATCH --job-name=dgm_i5_test
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --exclude=soda-03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:20:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

export PYTHONUNBUFFERED=1
source /workspace/chsong/miniconda3/bin/activate stylegan

export LD_LIBRARY_PATH=$(
python - <<'PY'
import site
from pathlib import Path

paths = []
for sp in site.getsitepackages():
    root = Path(sp) / "nvidia"
    if root.exists():
        for libdir in root.glob("*/lib"):
            if libdir.exists():
                paths.append(str(libdir))
print(":".join(paths))
PY
):$LD_LIBRARY_PATH

cd /workspace/chsong/dgm-face

echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

nvidia-smi

python - <<'PY'
import onnxruntime as ort
print("ORT version:", ort.__version__, flush=True)
print("ORT providers:", ort.get_available_providers(), flush=True)
assert "CUDAExecutionProvider" in ort.get_available_providers()
PY

python scripts/align_faces_insight.py \
  --input-dir /data/chsong/dgm-face/frames_raw_3f_0205080 \
  --out-dir /data/chsong/dgm-face/faces_256_insight_smoke \
  --meta-path /data/chsong/dgm-face/metadata/aligned_faces_insight_smoke.csv \
  --size 256 \
  --det-size 640,640 \
  --ctx-id 0 \
  --limit 100 \
  --no-sort \
  --flush-every 20 \
  --overwrite

echo "Smoke aligned count:"
find /data/chsong/dgm-face/faces_256_insight_smoke -name "*.png" | wc -l

echo "Done: $(date)"
